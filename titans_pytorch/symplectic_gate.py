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
        phase_mix: float = 0.0,
        phase_pairs: int | None = None
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
        self.phase_mix = phase_mix

        if self.top_k is not None and self.adaptive_topk_ratio is not None:
            raise ValueError("Only one of top_k or adaptive_topk_ratio can be set.")
        if not (0.0 <= self.phase_mix <= 1.0):
            raise ValueError("phase_mix must be in [0, 1].")

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

    def forward(self, x, return_moment_map = False, return_phase_map = False):
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
                if self.top_k is not None:
                    k = max(1, min(self.top_k, mag.shape[-1]))
                else:
                    strength = gate_pre.abs().mean().clamp(0.0, 1.0).item()
                    k = max(1, min(int(mag.shape[-1] * self.adaptive_topk_ratio * strength), mag.shape[-1]))

                topk = torch.topk(gate_pre, k = k, dim = -1).indices
                mask = torch.zeros_like(gate_pre, dtype = torch.bool)
                mask.scatter_(-1, topk, True)
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
            phase = self.to_phase(x)
            batch, seq_len, _ = phase.shape
            phase = phase.view(batch, seq_len, self.phase_pairs, 2)
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

        if return_moment_map:
             # Also pad the raw moment map
             momentum_map = F.pad(momentum_map, (0, 0, 1, 0), value=0.0)
             if return_phase_map:
                 return complexity_score, momentum_map, phase_score
             return complexity_score, momentum_map

        if return_phase_map:
            return complexity_score, phase_score

        return complexity_score
