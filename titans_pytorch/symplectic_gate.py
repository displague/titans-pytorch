import torch
import torch.nn as nn
import torch.nn.functional as F


class SymplecticGating(nn.Module):
    """
    Detects topological 'twists' (non-commutativity) in the input stream.
    Acts as the 'Cohomology Check' from the Iritani Proof.
    """

    def __init__(self, dim):
        super().__init__()
        # Projections to map input space to 'Twist Space'
        self.to_twist_q = nn.Linear(dim, dim, bias=False)
        self.to_twist_k = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.to_twist_q(x)
        k = self.to_twist_k(x)

        # Normalize to Unit Norm to make the wedge product area meaningful (sin theta)
        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)

        # The Wedge Product Approximation:
        # Measures the area spanned by consecutive vectors.
        # 0 Area = Rational/Linear line.
        # High Area = Loop/Cycle detected.

        q_curr, q_prev = q[:, 1:], q[:, :-1]
        k_curr, k_prev = k[:, 1:], k[:, :-1]

        # Symplectic Area Calculation: (q_i * k_{i-1}) - (q_{i-1} * k_i)
        twist = (q_curr * k_prev) - (q_prev * k_curr)

        # Norm gives us the magnitude of the twist
        twist_magnitude = twist.norm(dim=-1, keepdim=True)

        # Tanh normalizes this to a 0-1 "Complexity Score"
        # unlike sigmoid, tanh(0) = 0, so rational data has 0 complexity.
        complexity_score = torch.tanh(twist_magnitude)

        # Pad to match sequence length (prepend 0 to match input length)
        return F.pad(complexity_score, (0, 0, 1, 0), value=0.0)
