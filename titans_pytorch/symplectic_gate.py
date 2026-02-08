import torch
import torch.nn as nn
import torch.nn.functional as F


class SymplecticGating(nn.Module):
    """
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

    When `diag=True`, the gate and magnitude projections are constrained to diagonal
    (elementwise) scaling for a "diagonal tensor" proxy.
    """

    def __init__(
        self,
        dim,
        gated: bool = False,
        diag: bool = False,
        gate_threshold: float = 0.0
    ):
        super().__init__()
        # Projections to map input space to 'Twist Space' (Lie Algebra dual $\mathfrak{g}^*$ approximation)
        self.to_twist_q = nn.Linear(dim, dim, bias=False)
        self.to_twist_k = nn.Linear(dim, dim, bias=False)
        self.gated = gated
        self.diag = diag
        self.gate_threshold = gate_threshold

        if self.gated:
            if self.diag:
                self.gate_weight = nn.Parameter(torch.ones(dim))
                self.gate_bias = nn.Parameter(torch.zeros(dim))
                self.mag_weight = nn.Parameter(torch.ones(dim))
                self.mag_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.to_gate = nn.Linear(dim, dim)
                self.to_mag = nn.Linear(dim, dim)

    def forward(self, x, return_moment_map = False):
        if self.gated:
            if self.diag:
                gate_pre = x * self.gate_weight + self.gate_bias
                mag_pre = x * self.mag_weight + self.mag_bias
            else:
                gate_pre = self.to_gate(x)
                mag_pre = self.to_mag(x)

            gate_hard = (gate_pre > self.gate_threshold).to(x.dtype)
            gate_soft = torch.sigmoid(gate_pre)
            gate = gate_hard + (gate_soft - gate_soft.detach())

            mag = F.relu(mag_pre)
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

        if return_moment_map:
             # Also pad the raw moment map
             momentum_map = F.pad(momentum_map, (0, 0, 1, 0), value=0.0)
             return complexity_score, momentum_map

        return complexity_score
