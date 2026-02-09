from __future__ import annotations
from typing import Callable

import math
from functools import partial
from itertools import zip_longest
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from tensordict import TensorDict

from assoc_scan import AssocScan

from titans_pytorch.memory_models import(
    MemoryMLP,
    ResidualNorm
)
from titans_pytorch.symplectic_gate import SymplecticGating
from titans_pytorch.dmd import DynamicModeDecomposition

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

LinearNoBias = partial(Linear, bias = False)

# neural mem state related

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
    'active_page_indices'
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    state = tree_map(lambda t: t.detach() if is_tensor(t) else t, tuple(state))
    return NeuralMemState(*state)

# functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

# softclamping gradients

def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t

    t, inverse = pack_one_with_inverse(t, 'bn *')

    norm = t.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)

# spectral norming the surprise update w/ newton schulz matrix iter
# Keller Jordan et al. from OSS w/ nanogpt, now being used for two works, Atlas and 'TTT done right'

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)

# multi head rmsnorm

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# chunk pooling

class AveragePool(Module):
    def __init__(
        self,
        chunk_size
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim = -2)

        return reduce(x * attn, 'b n c d -> b n d', 'sum')

# main neural memory

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        batch_size = None,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        per_parameter_lr_modulation = False, # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation = 1., # max of 10.
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1, # whether a single token can do multiple updates to the memory model
        qkv_receives_diff_views = False, # to address an issue raised by a phd student (who will be credited if experiments are green). basically the issue raised is that the memory MLP is only learning Wk @ Wv linear mapping and that may not be expressive enough. we will use hyper connections to allow the network to choose different previous layer inputs as keys / values and see if that does anything
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        spectral_norm_surprises = False,
        gated_transition = False,
        mem_model_norm_add_residual = True, # by default, layernorm output and add residual as proposed in TTT paper,
        use_symplectic_gating = False, # New argument
        symplectic_gate_kwargs: dict | None = None, # Optional kwargs forwarded to SymplecticGating
        use_dmd_gating = False, # New argument
        combine_symplectic_and_dmd = False, # Optional: blend both gating signals when both are enabled
        manifold_state_keyed_paging = False, # Optional: route paging from manifold state keys when available
        hierarchical_paging = False, # Optional: two-stage coarse/fine page routing
        coarse_pages: int | None = None, # Optional: number of coarse routing groups
        fine_pages: int | None = None, # Optional: pages per coarse group
        hierarchy_mix: float = 1.0, # 0 = sequential fallback, 1 = fully hierarchical routing
        kinetics_coupling = False, # Optional: chemistry-inspired coupling between adaptive lr and decay
        kinetics_mix: float = 0.0, # Blend strength for kinetics coupling
        kinetics_eps: float = 1e-6, # Numerical stability for kinetics normalization
        num_pages = 1, # New argument: Manifold Paging for Objective Reduction
        symplectic_page_threshold: float | None = None, # Optional override for page switch threshold
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )
    ):
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.num_pages = num_pages
        # Effectively multiply heads by pages.
        # Each "Head" in the original sense now has 'num_pages' versions of itself.
        # We route updates to only one of them at a time.
        self.internal_heads = heads * num_pages
        self.user_heads = heads # Keep track of logical heads
        
        # We keep 'heads' variable as 'internal_heads' for creating sub-modules that need the expanded dim
        # But 'self.heads' must reflect USER intent (splitting input).
        
        heads_for_inner = self.internal_heads

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # batch size

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)

        self.batch_size = batch_size

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        # key values receiving different views

        self.qkv_receives_diff_views = qkv_receives_diff_views

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # use internal heads for any per-head normalization
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads_for_inner) if post_rmsnorm else nn.Identity()

        self.q_norm = MultiheadRMSNorm(dim_head, heads_for_inner) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads_for_inner) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * self.internal_heads

        # for internal tensor shaping, always use the expanded head count (pages * user heads)
        self.heads = self.internal_heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads_for_inner)
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads_for_inner, u = num_kv_per_token)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if self.internal_heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads_for_inner),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if self.internal_heads > 1 else None

        # memory model

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        # validate memory model

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')

            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

        # the memory is the weights of the model

        if mem_model_norm_add_residual:
            model = ResidualNorm(dim = dim_head, model = model)

        self.memory_model = model

        mem_model_params = dict(model.named_parameters())

        self.num_memory_parameter_tensors = len(mem_model_params)

        self.memory_model_parameter_names = [*mem_model_params.keys()]

        memory_model_parameters = [*mem_model_params.values()]

        if per_head_learned_parameters:
            # Expand to internal_heads by allocating distinct Parameter copies per head
            expanded_params = []
            for p in memory_model_parameters:
                stacked = torch.stack([p.detach().clone() for _ in range(heads_for_inner)], dim=0)
                expanded_params.append(nn.Parameter(stacked))
            memory_model_parameters = expanded_params

        self.init_weight_shape = [p.shape for p in memory_model_parameters]

        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        # two functions

        grad_fn = grad(forward_and_loss, has_aux = True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        assert num_kv_per_token > 0

        self.to_keys = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.to_values = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.store_memory_loss_fn = store_memory_loss_fn

        self.num_kv_per_token = num_kv_per_token

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads_for_inner * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # momentum related

        self.to_momentum = Sequential(
            nn.Linear(dim, heads_for_inner * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads_for_inner * momentum_order),
                Rearrange('b n (h o) -> o (b h) n', h = heads_for_inner),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads_for_inner * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads_for_inner),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # learned weight residual

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads_for_inner),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # spectral norming the surprises before update, a la Muon from Jordan et al.

        self.spectral_norm_surprises = spectral_norm_surprises

        # weight decay factor

        self.to_decay_factor = Sequential(
            Linear(dim, heads_for_inner),
            Rearrange('b n h -> (b h) n 1')
        )

        # Symplectic Gate Initialization
        self.use_symplectic_gating = use_symplectic_gating
        if use_symplectic_gating:
            symplectic_gate_kwargs = default(symplectic_gate_kwargs, {})
            self.symplectic_gate = SymplecticGating(dim, **symplectic_gate_kwargs)
            self.symplectic_complexity_scale = Parameter(torch.ones(1) * 0.5)
            
            # For Manifold Paging: Track the active page index
            # REMOVED global buffer 'active_page_index' in favor of per-sample state in NeuralMemState
            # Track number of page switches for logging
            self.register_buffer('page_switch_events', torch.zeros(1, dtype=torch.long))
            
            # Threshold for "Objective Reduction" (Collapsing to a new page)
            # This can be learned or fixed. Let's make it a parameter initialized to high value (0.7??)
            # Actually, let's make it fixed for now to test stability.
            self.symplectic_page_threshold = default(symplectic_page_threshold, 0.5)

        self.use_dmd_gating = use_dmd_gating
        self.combine_symplectic_and_dmd = combine_symplectic_and_dmd
        self.manifold_state_keyed_paging = manifold_state_keyed_paging
        self.hierarchical_paging = hierarchical_paging
        self.coarse_pages = default(coarse_pages, 1)
        self.fine_pages = default(fine_pages, self.num_pages)
        self.hierarchy_mix = hierarchy_mix
        self.kinetics_coupling = kinetics_coupling
        self.kinetics_mix = kinetics_mix
        self.kinetics_eps = kinetics_eps

        if not (0.0 <= self.hierarchy_mix <= 1.0):
            raise ValueError("hierarchy_mix must be in [0, 1].")
        if not (0.0 <= self.kinetics_mix <= 1.0):
            raise ValueError("kinetics_mix must be in [0, 1].")
        if self.kinetics_eps <= 0.0:
            raise ValueError("kinetics_eps must be > 0.")
        if self.hierarchical_paging:
            if self.num_pages <= 1:
                raise ValueError("hierarchical_paging requires num_pages > 1.")
            if self.coarse_pages < 1 or self.fine_pages < 1:
                raise ValueError("coarse_pages and fine_pages must be >= 1.")
            if (self.coarse_pages * self.fine_pages) != self.num_pages:
                raise ValueError("coarse_pages * fine_pages must equal num_pages when hierarchical_paging is enabled.")

        if use_dmd_gating:
             # Rank=None implies full rank SVD
             self.dmd = DynamicModeDecomposition(rank=None)
             # DMD complexity can be high, so we might want a different scale init?
             self.dmd_complexity_scale = Parameter(torch.ones(1) * 0.5)
             
             if not hasattr(self, 'page_switch_events'):
                 self.register_buffer('page_switch_events', torch.zeros(1, dtype=torch.long))

             # Default threshold for DMD might need tuning (DMD error is ~0.5 for random)
             self.symplectic_page_threshold = default(symplectic_page_threshold, 0.4)

        if self.manifold_state_keyed_paging and not hasattr(self, 'page_switch_events'):
            self.register_buffer('page_switch_events', torch.zeros(1, dtype=torch.long))


        # learned transition, as seeing instability when decreasing neural mem batch size
        # perhaps it can slowly learn to adjust from early residual to fully transitioning to new weights every batch size

        self.transition_gate = nn.Parameter(tensor(-5.)) if gated_transition else None

        # inits

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(
        self,
        batch,
    ):
        # We must init weights for ALL internal heads (User * Pages)
        # self.heads tracks User Heads. self.internal_heads tracks Total Heads. (See init)
        
        if self.per_head_learned_parameters:
            # Parameters already have 'h' dim = internal_heads
            weights = repeat_dict_values(self.memory_model_parameter_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch * self.internal_heads)

        return weights

    def init_momentum(
        self,
        batch,
    ):
        zeros = self.memory_model_parameter_dict.clone().zero_()

        if self.per_head_learned_parameters:
            zeros = repeat_dict_values(zeros, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = repeat_dict_values(zeros, '... -> o bh ...', bh = batch * self.internal_heads, o = self.momentum_order)

        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True,
        active_page_indices: Tensor | None = None
    ):
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        # shapes and variables

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]

        next_seq_len_index = seq_index + round_down_seq_len

        # init weights if needed
        # weights of the memory network

        if not exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)

        # allow for neural memory of a previous layer to influence surprise of current layer

        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)

        # initial norm

        seq = self.store_norm(seq)

        # handle keys and values coming from different sequences from hyper connection

        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        if self.use_symplectic_gating or self.use_dmd_gating:
             # Calculate Complexity
             use_combined = self.combine_symplectic_and_dmd and self.use_symplectic_gating and self.use_dmd_gating
             needs_manifold_state = self.manifold_state_keyed_paging or self.hierarchical_paging
             manifold_state = None

             if use_combined:
                 # Symplectic signal: local geometric twist.
                 if needs_manifold_state:
                     symplectic_complexity, manifold_state = self.symplectic_gate(seq, return_manifold_state = True)
                 else:
                     symplectic_complexity = self.symplectic_gate(seq)
                 # DMD signal: global linear-dynamics reconstruction error.
                 dmd_error = self.dmd(seq)
                 dmd_complexity = torch.tanh(dmd_error).view(-1, 1, 1).expand(-1, seq.shape[1], 1)

                 complexity = 0.5 * symplectic_complexity + 0.5 * dmd_complexity
                 scale_param = 0.5 * (self.symplectic_complexity_scale + self.dmd_complexity_scale)

             elif self.use_symplectic_gating:
                 # (B, Seq, 1)
                 if needs_manifold_state:
                     complexity, manifold_state = self.symplectic_gate(seq, return_manifold_state = True)
                 else:
                     complexity = self.symplectic_gate(seq)
                 scale_param = self.symplectic_complexity_scale

             elif self.use_dmd_gating:
                 # (B,) -> Broadcast to (B, Seq, 1)
                 # DMD gives global error for the window. We normalize it.
                 dmd_error = self.dmd(seq)
                 # Tanh to squash to [0,1]
                 complexity = torch.tanh(dmd_error).view(-1, 1, 1).expand(-1, seq.shape[1], 1)
                 scale_param = self.dmd_complexity_scale
             
             # Reduce to chunked representation to match decay_factor slope
             # CRITICAL: Use MAX reduction instead of self.reduce_to_chunk_rep (Average/Attn).
             # We want to detect if *any* part of the chunk has a twist. Averaging washes out fine-grained structure.
             complexity_chunked = reduce(complexity, 'b (n c) d -> b n d', 'max', c = chunk_size)
             
             # Expand dimensions if necessary to match (B, N, H, 1)
             complexity_chunked = complexity_chunked.unsqueeze(-1)
             
             # Reshape decay_factor to (B, N, H, 1) to mix properly
             # Currently it is (B*H, N, 1) from to_decay_factor
             decay_factor = rearrange(decay_factor, '(b h) n 1 -> b n h 1', h = heads)
             # Adaptive Decay Logic:
             # High Complexity -> High Retention (Low Decay)
             # decay_new = decay_old * (1 - scale * complexity)
             decay_factor = decay_factor * (1. - scale_param * complexity_chunked)

             # Reshape back to (B*H, N, 1)
             decay_factor = rearrange(decay_factor, 'b n h 1 -> (b h) n 1')


             # --- MANIFOLD PAGING (OBJECTIVE REDUCTION) ---
             if self.num_pages > 1:
                 # Initialize active_page_indices if None (First chunk)
                 if not exists(active_page_indices):
                     active_page_indices = torch.zeros(batch, dtype=torch.long, device=seq.device)

                 # Check if we need to undergo Objective Reduction (Page Switch)
                 # We look at the max complexity in the chunk per sample.
                 # complexity: (B, ChunkLen, 1)
                 # max_complexity per sample: (B, 1)
                 max_complexity_per_sample = reduce(complexity, 'b n 1 -> b 1', 'max')
                 
                 # Determine which samples need to switch page
                 should_switch = (max_complexity_per_sample > self.symplectic_page_threshold).squeeze(-1) # (B,)
                 
                 # Increment logging counter (approximate, counts events not total switches)
                 if should_switch.any():
                     self.page_switch_events.add_(should_switch.sum())

                 sequential_pages = (active_page_indices + 1) % self.num_pages
                 keyed_pages = sequential_pages
                 manifold_pages = None

                 if exists(manifold_state):
                     phase_angle = manifold_state.get('phase_angle', None)
                     if exists(phase_angle):
                         mean_sin = phase_angle.sin().mean(dim = (-1, -2))
                         mean_cos = phase_angle.cos().mean(dim = (-1, -2))
                         mean_angle = torch.atan2(mean_sin, mean_cos)
                         normalized = ((mean_angle + math.pi) / (2 * math.pi)).clamp(min = 0., max = 0.999999)
                         fine_page_count = self.fine_pages if self.hierarchical_paging else self.num_pages
                         manifold_pages = (normalized * fine_page_count).long().clamp(max = fine_page_count - 1)

                 if self.hierarchical_paging:
                     coarse_scores = max_complexity_per_sample.squeeze(-1).clamp(min = 0., max = 0.999999)
                     coarse_index = (coarse_scores * self.coarse_pages).long().clamp(max = self.coarse_pages - 1)

                     fallback_fine = sequential_pages % self.fine_pages
                     fine_index = manifold_pages if exists(manifold_pages) else fallback_fine
                     hierarchical_pages = (coarse_index * self.fine_pages + fine_index).clamp(max = self.num_pages - 1)

                     if self.hierarchy_mix <= 0.0:
                         keyed_pages = sequential_pages
                     elif self.hierarchy_mix >= 1.0:
                         keyed_pages = hierarchical_pages
                     else:
                         blended = (
                             (1. - self.hierarchy_mix) * sequential_pages.float()
                             + self.hierarchy_mix * hierarchical_pages.float()
                         )
                         keyed_pages = blended.round().long().clamp(min = 0, max = self.num_pages - 1)
                 elif self.manifold_state_keyed_paging and exists(manifold_pages):
                     keyed_pages = manifold_pages

                 # Switch pages for the samples that triggered the threshold.
                 active_page_indices = torch.where(should_switch, keyed_pages, active_page_indices)

                 # Update active_page_indices for next state return
                 # (It stays as tensor for now)

                 # Create Page Mask
                 # We need to construct a mask of shape (B, Total_Heads)
                 # where Total_Heads = User_Heads * Num_Pages.
                 # For each sample b, only heads [page[b]*UserH : (page[b]+1)*UserH] should be active.
                 
                 # Create a range [0, 1, ..., InternalHeads-1]
                 head_indices = torch.arange(self.internal_heads, device=seq.device).unsqueeze(0) # (1, TotalH)
                 
                 # Determine valid ranges per sample
                 # min_valid = page[b] * UserH
                 # max_valid = (page + 1) * UserH
                 min_valid = (active_page_indices * self.user_heads).unsqueeze(1) # (B, 1)
                 max_valid = ((active_page_indices + 1) * self.user_heads).unsqueeze(1) # (B, 1)
                 
                 # Mask is 1 where min_valid <= index < max_valid
                 page_mask = (head_indices >= min_valid) & (head_indices < max_valid) # (B, TotalH)
                 page_mask = page_mask.float()
                 
                 # Apply mask to adaptive_lr
                 # adaptive_lr comes from to_adaptive_step: (B*H_total, N*U)
                 # Reshape to (B, H_total, N*U)
                 adaptive_lr = rearrange(adaptive_lr, '(b h) nu -> b h nu', h=self.internal_heads)
                 
                 # Apply mask
                 adaptive_lr = adaptive_lr * page_mask.unsqueeze(-1)
                 
                 # Reshape back
                 adaptive_lr = rearrange(adaptive_lr, 'b h nu -> (b h) nu')

                 # CRITICAL FIX: Also mask decay_factor!
                 # If we don't, inactive pages will decay (gate < 1) but receive 0 update, causing forgetting.
                 # We want decay=0 (gate=1) for inactive pages to preserve memory exactly.
                 
                 # decay_factor is (B*H, N, 1). Reshape to (B, H, N, 1)
                 decay_factor = rearrange(decay_factor, '(b h) n 1 -> b h n 1', h=self.internal_heads)
                 
                 # Apply mask. page_mask is (B, H).
                 decay_factor = decay_factor * page_mask.unsqueeze(-1).unsqueeze(-1)
                 
                 # Reshape back
                 decay_factor = rearrange(decay_factor, 'b h n 1 -> (b h) n 1')



        # --- CHEMISTRY-INSPIRED KINETICS COUPLING ---
        # Couple update and decay rates via a bounded "reaction progress" proxy.
        # Higher update pressure pushes toward higher adaptive lr and lower decay.
        if self.kinetics_coupling and self.kinetics_mix > 0.0:
            adaptive_lr_chunk = rearrange(adaptive_lr, 'bh (n c u) -> bh n (c u)', c = chunk_size, u = num_updates)
            adaptive_lr_mean = adaptive_lr_chunk.mean(dim = -1, keepdim = True)

            reaction_progress = adaptive_lr_mean / (adaptive_lr_mean + decay_factor + self.kinetics_eps)
            decay_factor = decay_factor * (1. - self.kinetics_mix * reaction_progress)
            decay_factor = decay_factor.clamp(min = 0., max = 1.)

            reaction_progress_tokens = repeat(
                reaction_progress,
                'bh n 1 -> bh n cu',
                cu = chunk_size * num_updates
            )
            reaction_progress_tokens = rearrange(reaction_progress_tokens, 'bh n cu -> bh (n cu)')
            adaptive_lr = adaptive_lr + self.kinetics_mix * (1. - adaptive_lr) * reaction_progress_tokens
            adaptive_lr = adaptive_lr.clamp(min = 0.)

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # keys and values

        keys = self.to_keys(seq)
        values = self.to_values(values_seq)

        # maybe multi head

        keys, values = map(self.split_kv_heads, (keys, values))

        # maybe keys rmsnorm

        keys = self.k_norm(keys)

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates) for t in (keys, values))

        # adaptive lr

        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # optionally a storing memories mask can be passed in. if False, will set the learning rate to 0. for those positions

        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)

            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # maybe add previous layer weight

        assert xnor(exists(self.to_learned_weight_residual_mix), exists(prev_weights))

        if exists(prev_weights):

            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks

            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights = prev_weights.apply(lambda t: einx.multiply('bh n, bh n ... -> bh n ...', mix, t))

            weights_for_surprise = weights_for_surprise + prev_weights

        # flatten batch and time for per-sample gradient mapping
        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # surprises

        adaptive_lr = rearrange(adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.mul(-1)

        # past states

        if not exists(past_state):
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper

            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)

            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early return if sequence length less than chunk size

        if num_chunks == 0:
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates, active_page_indices)

            output = (updates, next_store_state)

            if not return_surprises:
                return output

            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                momentum = surprise

                momentums = [] # stores all momentum orders starting with first, to generalize to Nth order momentum

                last_momentum = past_last_momentum[param_name]

                # go from first order momentum all the way to the Nth

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum) # momentum is S / surprise in the paper

                    momentums.append(momentum)

                momentums = stack(momentums)

                next_last_momentum[param_name] = momentums[:, :, -1] # momentums shape is Float['o bh n 1']

                if learned_combine and self.learned_combine_include_zeroth:
                    # add the original surprise if learned combination of momentums
                    momentums = cat((rearrange(surprise, '... -> 1 ...'), momentums), dim = 0)

                if not learned_combine:
                    update = momentums[-1]
                else:
                    update = einsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')

            # maybe spectral norm surprises

            if self.spectral_norm_surprises:
                update = newtonschulz5(update)

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)

            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        next_state = (next_last_update, next_last_momentum)

        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates, active_page_indices)

        # return updates to neural memory at all chunked timesteps + neural mem cache / state to be fed back

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: dict[str, Tensor],
    ):
        chunk_size = self.retrieve_chunk_size

        weights_have_expanded_shape = dict_get_value_shapes(weights) != self.init_weight_shape

        batch, seq_len = seq.shape[:2]

        # auto infer single token decoding, if there are only 1 set of weights and 1 token

        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1

        is_single_token_decode = is_one_token and is_one_weight

        if is_single_token_decode:
            chunk_size = 1

        # padding related, for chunked processing

        need_pad = chunk_size > 1 or not is_one_weight

        if need_pad:
            seq = pad_at_dim(seq, (1, 0), dim = 1)

        seq_len_plus_one = seq.shape[-2]

        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)

        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        weights = TensorDict(weights)

        # pre norm

        seq = self.retrieve_norm(seq)

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

        # fetch values from memory model

        if weights_have_expanded_shape:
            weights = rearrange_dict_values(weights, 'b n ... -> (b n) ...')

        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

        # forward functional call

        values = functional_call(self.memory_model, dict(weights), queries)

        # reconstitute batch dimension

        values = rearrange(values, '(b h n) c d -> b h (n c) d', b = batch, h = self.heads)

        values = self.multihead_rmsnorm(values)

        # maybe gate

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine

        values = self.merge_heads(values)

        values = self.combine_heads(values)

        # restore, pad with empty memory embed

        if need_pad:
            values = values[:, 1:]

        return values[:, :seq_len]

    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None
    ):
        is_multi_input = self.qkv_receives_diff_views

        # handle single token

        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, '... b d -> ... b 1 d')

        is_single_token = seq.shape[-2] == 1

        # if different views for qkv, then

        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        # handle previous state init

        # handle previous state init

        if not exists(state):
            state = (0, None, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates, active_page_indices = state

        # store

        store_seq = default(store_seq, seq)

        # take care of cache

        if exists(cache_store_seq):
            store_seq = safe_cat((cache_store_seq, store_seq))

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, default(ttt_batch_size, self.batch_size)

        need_update_weights = exists(batch_size)

        # determine split sizes and when to update

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)

            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = divisible_by(seq_range, batch_size)

            indices = seq_range[batch_boundary] - seq_index

            indices = F.pad(indices, (1, 0), value = 0)

            if indices[-1] != store_seq_len:
                indices = F.pad(indices, (0, 1), value = store_seq_len)

            split_sizes = (indices[1:] - indices[:-1]).tolist()

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        # accumulate updates

        updates = None

        def accum_updates(past_updates, future_updates):
            if not exists(past_updates):
                return future_updates

            return TensorDict({param_name: cat((past_update[:, :-1], future_update), dim = 1) for (param_name, past_update), (_, future_update) in zip(past_updates.items(), future_updates.items())})

        # loop through chunks of store sequences

        store_seqs = store_seq.split(split_sizes, dim = -2)

        if exists(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)

        # whether to allow network to slowly adjust from initial weight throughout (residual path) to fully updating weights every batch

        surprises = (None, None)
        gate = None

        if exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index = seq_index,
                past_state = past_state,
                prev_weights = prev_weights,
                mask = maybe_store_mask,
                return_surprises = True,
                active_page_indices = active_page_indices
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states
            active_page_indices = next_neural_mem_state.active_page_indices

            updates = accum_updates(updates, next_updates)

            surprises = tuple(safe_cat(args, dim = -1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            # update weights once batch size is fulfilled

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = TensorDict({param_name: one_weight.lerp(one_last_update, gate) for (param_name, one_weight), (_, one_last_update) in zip(weights.items(), last_update.items())})

            past_state = (last_update, last_momentum)

            # set weights to the last updated weights for the last minibatch

            weights = last_update

            next_neural_mem_state = next_neural_mem_state._replace(
                weights = weights,
                states = past_state,
            )

        next_neural_mem_state = next_neural_mem_state._replace(updates = updates)

        # retrieve

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = rearrange_dict_values(last_update, 'b ... -> b 1 ...')

        retrieved = self.retrieve_memories(
            retrieve_seq,
            updates
        )

        # maybe detach

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
