"""
Quantile network: sorted standardized order statistics -> monotone quantiles
of the standardized root.

Output head: q(tau_1) = base, q(tau_k) = base + cumsum(softplus(increments)).
Monotonicity is enforced by construction, there are no shape restrictions
(handles hard boundaries like the uniform max, where a GMM head fails), and
the predicted quantiles directly yield confidence intervals.

The input is a fixed-length sorted vector, so a plain MLP is a valid set
encoder here (sorting is the canonical representation of an exchangeable
sample). A variable-n set encoder is a later-milestone swap.
"""

import torch
import torch.nn as nn


class QuantileNet(nn.Module):

    def __init__(self, n_input: int = 200, n_aux: int = 2,
                 n_levels: int = 199, hidden: int = 256, depth: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        d = n_input + n_aux
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_base = nn.Linear(hidden, 1)
        self.head_incr = nn.Linear(hidden, n_levels - 1)

    def forward(self, z: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z   (B, n)  sorted standardized values
            aux (B, 2)  [log s, med/s]
        Returns:
            q   (B, n_levels)  monotone quantiles of the standardized root
        """
        h = self.trunk(torch.cat([z, aux], dim=1))
        base = self.head_base(h)                              # (B, 1)
        incr = nn.functional.softplus(self.head_incr(h))      # (B, L-1)
        return torch.cat([base, base + torch.cumsum(incr, dim=1)], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
