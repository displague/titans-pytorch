import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymplecticGating(nn.Module):
    r"""
    Detects topological 'twists' (non-commutativity) in the input stream.
    
    This module implements a localized version of the **Moment Map** ($\mu$) from the 
    Marsden-Weinstein-Meyer Theorem (Symplectic Reduction).
    
    It measures the failure of the "Rationality" condition (Lagrangian Submanifold condition)
    by computing the symplectic area spanned by consecutive transitions in the latent space.
    
    High Moment Map Magnitude (|$\mu$| >> 0) indicates a "Twist" or topological feature 
    that breaks the symmetry of the rational flow, requiring "Objective Reduction" 
    (Manifold Paging) to resolve.

    Optional gated SAE-style projection (toggle via `gated=True`) is inspired by:
    - "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic)
    - "Improving Dictionary Learning with Gated Sparse Autoencoders" (Anthropic)
    Optional adaptive sparsity (`top_k` / `adaptive_topk_ratio`) is aligned with AdaptiveK-style routing.
    Optional phase-aware complexity (`phase_mix > 0`) captures periodic latent dynamics
    using per-step angular change in paired latent channels (closed-loop / limit-cycle proxy).
    Optional quorum/budget policy (`quorum_mix > 0`) applies local consensus smoothing and
    optional sequence-budgeted selection on complexity scores.
    Optional codebook policy (`codebook_mix > 0`) models combinatorial receptor coding by
    tracking shifts in sparse codebook assignments over time.

    When `diag=True`, the gate and magnitude projections are constrained to diagonal
    (elementwise) scaling for a "diagonal tensor" proxy.
    """

    def __init__(
        self,
        dim,
        gated: bool = False,
        diag: bool = False,
        gate_threshold: float = 0.0,
        gate_mode: str = "hard",
        top_k: int | None = None,
        adaptive_topk_ratio: float | None = None,
        adaptive_topk_min_k: int = 1,
        phase_mix: float = 0.0,
        phase_pairs: int | None = None,
        quorum_mix: float = 0.0,
        quorum_window: int = 3,
        quorum_threshold: float = 0.5,
        quorum_temperature: float = 0.1,
        budget_topk_ratio: float | None = None,
        codebook_mix: float = 0.0,
        codebook_size: int = 16,
        codebook_temperature: float = 0.1,
        codebook_topk: int | None = None
    ):
        super().__init__()
        # Projections to map input space to 'Twist Space' (Lie Algebra dual $\mathfrak{g}^*$ approximation)
        self.to_twist_q = nn.Linear(dim, dim, bias=False)
        self.to_twist_k = nn.Linear(dim, dim, bias=False)
        self.gated = gated
        self.diag = diag
        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.top_k = top_k
        self.adaptive_topk_ratio = adaptive_topk_ratio
        self.adaptive_topk_min_k = adaptive_topk_min_k
        self.phase_mix = phase_mix
        self.quorum_mix = quorum_mix
        self.quorum_window = quorum_window
        self.quorum_threshold = quorum_threshold
        self.quorum_temperature = quorum_temperature
        self.budget_topk_ratio = budget_topk_ratio
        self.codebook_mix = codebook_mix
        self.codebook_size = codebook_size
        self.codebook_temperature = codebook_temperature
        self.codebook_topk = codebook_topk

        if self.top_k is not None and self.adaptive_topk_ratio is not None:
            raise ValueError("Only one of top_k or adaptive_topk_ratio can be set.")
        if self.adaptive_topk_min_k < 1:
            raise ValueError("adaptive_topk_min_k must be >= 1.")
        if self.adaptive_topk_ratio is not None and not (0.0 < self.adaptive_topk_ratio <= 1.0):
            raise ValueError("adaptive_topk_ratio must be in (0, 1].")
        if not (0.0 <= self.phase_mix <= 1.0):
            raise ValueError("phase_mix must be in [0, 1].")
        if not (0.0 <= self.quorum_mix <= 1.0):
            raise ValueError("quorum_mix must be in [0, 1].")
        if self.quorum_window < 1:
            raise ValueError("quorum_window must be >= 1.")
        if self.quorum_temperature <= 0.0:
            raise ValueError("quorum_temperature must be > 0.")
        if self.budget_topk_ratio is not None and not (0.0 < self.budget_topk_ratio <= 1.0):
            raise ValueError("budget_topk_ratio must be in (0, 1].")
        if self.budget_topk_ratio is not None and self.quorum_mix <= 0.0:
            raise ValueError("budget_topk_ratio requires quorum_mix > 0.")
        if not (0.0 <= self.codebook_mix <= 1.0):
            raise ValueError("codebook_mix must be in [0, 1].")
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2.")
        if self.codebook_temperature <= 0.0:
            raise ValueError("codebook_temperature must be > 0.")
        if self.codebook_topk is not None and self.codebook_topk < 1:
            raise ValueError("codebook_topk must be >= 1.")
        if self.codebook_topk is not None and self.codebook_topk > self.codebook_size:
            raise ValueError("codebook_topk must be <= codebook_size.")

        if self.gated:
            if self.diag:
                self.gate_weight = nn.Parameter(torch.ones(dim))
                self.gate_bias = nn.Parameter(torch.zeros(dim))
                self.mag_weight = nn.Parameter(torch.ones(dim))
                self.mag_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.to_gate = nn.Linear(dim, dim)
                self.to_mag = nn.Linear(dim, dim)

        self.phase_pairs = phase_pairs
        if self.phase_mix > 0.0:
            self.phase_pairs = phase_pairs if phase_pairs is not None else max(1, dim // 2)
            self.to_phase = nn.Linear(dim, self.phase_pairs * 2, bias = False)
            nn.init.normal_(self.to_phase.weight, std = 0.02)

        if self.codebook_mix > 0.0:
            self.to_codebook = nn.Linear(dim, self.codebook_size, bias = False)
            nn.init.normal_(self.to_codebook.weight, std = 0.02)

    def compute_sparse_mask(
        self,
        gate_pre: torch.Tensor,
        return_k: bool = False
    ):
        feature_dim = gate_pre.shape[-1]
        min_k = min(self.adaptive_topk_min_k, feature_dim)

        if self.top_k is None and self.adaptive_topk_ratio is None:
            mask = torch.ones_like(gate_pre, dtype = torch.bool)
            if return_k:
                k = torch.full((*gate_pre.shape[:-1], 1), feature_dim, dtype = torch.long, device = gate_pre.device)
                return mask, k
            return mask

        if self.top_k is not None:
            k = torch.full(
                (*gate_pre.shape[:-1], 1),
                max(min_k, min(self.top_k, feature_dim)),
                dtype = torch.long,
                device = gate_pre.device
            )
        else:
            # Per-token adaptive k based on activation strength, normalized with tanh to [0, 1).
            token_strength = gate_pre.abs().mean(dim = -1, keepdim = True).tanh()
            target = token_strength * (self.adaptive_topk_ratio * feature_dim)
            k = torch.ceil(target).long().clamp(min = min_k, max = feature_dim)

        rank_order = torch.argsort(gate_pre, dim = -1, descending = True)
        ranks = torch.empty_like(rank_order)
        arange = torch.arange(feature_dim, device = gate_pre.device).view(*([1] * (gate_pre.ndim - 1)), feature_dim)
        ranks.scatter_(-1, rank_order, arange.expand_as(rank_order))

        mask = ranks < k

        if return_k:
            return mask, k
        return mask

    def extract_manifold_state(
        self,
        x: torch.Tensor
    ):
        if self.phase_mix <= 0.0:
            return None

        phase = self.to_phase(x)
        batch, seq_len, _ = phase.shape
        phase = phase.view(batch, seq_len, self.phase_pairs, 2)

        radius = phase.norm(dim = -1)
        angle = torch.atan2(phase[..., 1], phase[..., 0])

        return dict(
            phase_radius = radius,
            phase_angle = angle
        )

    def apply_quorum_policy(
        self,
        complexity_score: torch.Tensor,
        return_budget_k: bool = False
    ):
        quorum_score = torch.ones_like(complexity_score)
        budget_k = None

        if self.quorum_mix <= 0.0:
            if return_budget_k:
                return complexity_score, quorum_score, budget_k
            return complexity_score, quorum_score

        local_signal = complexity_score
        if self.quorum_window > 1:
            pooled = F.avg_pool1d(
                complexity_score.transpose(1, 2),
                kernel_size = self.quorum_window,
                stride = 1,
                padding = self.quorum_window // 2
            )
            # Keep exact sequence length for even/odd windows.
            local_signal = pooled[..., :complexity_score.shape[1]].transpose(1, 2)

        quorum_score = torch.sigmoid((local_signal - self.quorum_threshold) / self.quorum_temperature)

        if self.budget_topk_ratio is not None:
            seq_len = complexity_score.shape[1]
            budget_k = max(1, min(seq_len, math.ceil(self.budget_topk_ratio * seq_len)))
            quorum_rank = quorum_score.squeeze(-1)
            topk_indices = quorum_rank.topk(k = budget_k, dim = 1).indices
            budget_mask = torch.zeros_like(quorum_rank, dtype = torch.bool)
            budget_mask.scatter_(1, topk_indices, True)
            quorum_score = torch.where(budget_mask.unsqueeze(-1), quorum_score, torch.zeros_like(quorum_score))

        mix = (1.0 - self.quorum_mix) + (self.quorum_mix * quorum_score)
        complexity_score = complexity_score * mix

        if return_budget_k:
            return complexity_score, quorum_score, budget_k
        return complexity_score, quorum_score

    def compute_codebook_score(
        self,
        x: torch.Tensor
    ):
        if self.codebook_mix <= 0.0:
            return torch.zeros(*x.shape[:2], 1, device = x.device, dtype = x.dtype)

        logits = self.to_codebook(x) / self.codebook_temperature
        codebook_probs = logits.softmax(dim = -1)

        if self.codebook_topk is not None and self.codebook_topk < self.codebook_size:
            topk_indices = codebook_probs.topk(self.codebook_topk, dim = -1).indices
            mask = torch.zeros_like(codebook_probs, dtype = torch.bool)
            mask.scatter_(-1, topk_indices, True)
            codebook_probs = torch.where(mask, codebook_probs, torch.zeros_like(codebook_probs))
            codebook_probs = codebook_probs / codebook_probs.sum(dim = -1, keepdim = True).clamp(min = 1e-8)

        # Total variation distance between consecutive code distributions.
        # For probability distributions p, q: TV = 0.5 * ||p - q||_1 in [0, 1].
        code_prev = codebook_probs[:, :-1]
        code_curr = codebook_probs[:, 1:]
        codebook_score = 0.5 * (code_curr - code_prev).abs().sum(dim = -1, keepdim = True)
        codebook_score = F.pad(codebook_score, (0, 0, 1, 0), value = 0.0)
        return codebook_score

    def forward(
        self,
        x,
        return_moment_map = False,
        return_phase_map = False,
        return_codebook_map = False,
        return_sparse_k = False,
        return_manifold_state = False,
        return_quorum_map = False
    ):
        sparse_k = None
        manifold_state = None

        if self.gated:
            if self.diag:
                gate_pre = x * self.gate_weight + self.gate_bias
                mag_pre = x * self.mag_weight + self.mag_bias
            else:
                gate_pre = self.to_gate(x)
                mag_pre = self.to_mag(x)

            if self.gate_mode not in ("hard", "soft"):
                raise ValueError("gate_mode must be 'hard' or 'soft'")

            if self.gate_mode == "soft":
                gate = torch.sigmoid(gate_pre)
            else:
                gate_hard = (gate_pre > self.gate_threshold).to(x.dtype)
                gate_soft = torch.sigmoid(gate_pre)
                gate = gate_hard + (gate_soft - gate_soft.detach())

            mag = F.relu(mag_pre)
            if self.top_k is not None or self.adaptive_topk_ratio is not None:
                mask, sparse_k = self.compute_sparse_mask(gate_pre, return_k = True)
                gate = torch.where(mask, gate, torch.zeros_like(gate))
                mag = torch.where(mask, mag, torch.zeros_like(mag))

            x = gate * mag

        q = self.to_twist_q(x)
        k = self.to_twist_k(x)

        # Normalize to Unit Norm to make the wedge product area meaningful (sin theta)
        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)

        # The Wedge Product Approximation:
        # Measures the area spanned by consecutive vectors in the Phase Space.
        # $\mu = q_i \wedge k_{i-1} - q_{i-1} \wedge k_i$
        # 0 Area = Rational/Linear line (Flow stays on Lagrangian Submanifold).
        # High Area = Loop/Cycle detected (Symmetry Breaking / Non-Integrable).

        q_curr, q_prev = q[:, 1:], q[:, :-1]
        k_curr, k_prev = k[:, 1:], k[:, :-1]

        # Symplectic Area Calculation (Moment Map $\mu$)
        momentum_map = (q_curr * k_prev) - (q_prev * k_curr)

        # Norm gives us the magnitude of the twist $|\mu|$
        twist_magnitude = momentum_map.norm(dim=-1, keepdim=True)

        # Tanh normalizes this to a 0-1 "Complexity Score"
        # unlike sigmoid, tanh(0) = 0, so rational data (flat moment map) has 0 complexity.
        complexity_score = torch.tanh(twist_magnitude)

        # Pad to match sequence length (prepend 0 to match input length)
        complexity_score = F.pad(complexity_score, (0, 0, 1, 0), value=0.0)

        phase_score = torch.zeros_like(complexity_score)
        if self.phase_mix > 0.0:
            phase_state = self.extract_manifold_state(x)
            manifold_state = phase_state
            phase = torch.stack((phase_state["phase_radius"] * torch.cos(phase_state["phase_angle"]), phase_state["phase_radius"] * torch.sin(phase_state["phase_angle"])), dim = -1)
            phase = F.normalize(phase, dim = -1)

            phase_prev = phase[:, :-1]
            phase_curr = phase[:, 1:]

            cross = (phase_prev[..., 0] * phase_curr[..., 1]) - (phase_prev[..., 1] * phase_curr[..., 0])
            dot = (phase_prev * phase_curr).sum(dim = -1).clamp(min = -1., max = 1.)

            # |atan2(cross, dot)| / pi gives normalized angular step size in [0, 1].
            phase_delta = torch.atan2(cross, dot).abs() / torch.pi
            phase_score = phase_delta.mean(dim = -1, keepdim = True)
            phase_score = F.pad(phase_score, (0, 0, 1, 0), value = 0.0)

            complexity_score = (1. - self.phase_mix) * complexity_score + self.phase_mix * phase_score

        codebook_score = torch.zeros_like(complexity_score)
        if self.codebook_mix > 0.0:
            codebook_score = self.compute_codebook_score(x)
            complexity_score = (1. - self.codebook_mix) * complexity_score + self.codebook_mix * codebook_score

        complexity_score, quorum_score = self.apply_quorum_policy(complexity_score)

        if return_moment_map:
             # Also pad the raw moment map
             momentum_map = F.pad(momentum_map, (0, 0, 1, 0), value=0.0)
             if return_phase_map:
                 output = [complexity_score, momentum_map, phase_score]
                 if return_codebook_map:
                     output.append(codebook_score)
                 if return_quorum_map:
                     output.append(quorum_score)
                 if return_sparse_k:
                     output.append(sparse_k)
                 if return_manifold_state:
                     output.append(manifold_state)
                 return tuple(output)
             output = [complexity_score, momentum_map]
             if return_codebook_map:
                 output.append(codebook_score)
             if return_quorum_map:
                 output.append(quorum_score)
             if return_sparse_k:
                 output.append(sparse_k)
             if return_manifold_state:
                 output.append(manifold_state)
             return tuple(output)

        if return_phase_map:
            output = [complexity_score, phase_score]
            if return_codebook_map:
                output.append(codebook_score)
            if return_quorum_map:
                output.append(quorum_score)
            if return_sparse_k:
                output.append(sparse_k)
            if return_manifold_state:
                output.append(manifold_state)
            return tuple(output)

        if return_sparse_k or return_manifold_state or return_quorum_map or return_codebook_map:
            output = [complexity_score]
            if return_codebook_map:
                output.append(codebook_score)
            if return_quorum_map:
                output.append(quorum_score)
            if return_sparse_k:
                output.append(sparse_k)
            if return_manifold_state:
                output.append(manifold_state)
            return tuple(output)

        return complexity_score
