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
    """

    def __init__(self, dim):
        super().__init__()
        # Projections to map input space to 'Twist Space' (Lie Algebra dual $\mathfrak{g}^*$ approximation)
        self.to_twist_q = nn.Linear(dim, dim, bias=False)
        self.to_twist_k = nn.Linear(dim, dim, bias=False)

    def forward(self, x, return_moment_map = False):
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
