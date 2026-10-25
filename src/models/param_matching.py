"""Capacity matching: choose the MLP width so that its trainable-parameter count is within +-5 % of GraphSAGE.

Architectures (identical depth/width logic in the numpy and torch back-ends):
  GraphSAGE : SAGE(d->h) , SAGE(h->h) , Linear(h->c)      with separate self / neighbour weights
              params = (d*h + h + d*h) + (h*h + h + h*h) + (h*c + c) = 2dh + 2h^2 + 2h + hc + c
  MLP       : Linear(d->m), Linear(m->m), Linear(m->c)
              params = (d*m + m) + (m*m + m) + (m*c + c)
"""
from __future__ import annotations

from typing import Tuple


def sage_param_count(d: int, h: int, c: int = 2) -> int:
    return (2 * d * h + h) + (2 * h * h + h) + (h * c + c)


def mlp_param_count(d: int, m: int, c: int = 2) -> int:
    return (d * m + m) + (m * m + m) + (m * c + c)


def match_mlp_hidden(d: int, h: int, c: int = 2, tol: float = 0.05, max_width: int = 16384) -> Tuple[int, int, int]:
    """Return (mlp_hidden, mlp_params, sage_params) with relative difference <= tol (closest width)."""
    target = sage_param_count(d, h, c)
    best_m = min(range(1, max_width + 1), key=lambda m: abs(mlp_param_count(d, m, c) - target))
    p = mlp_param_count(d, best_m, c)
    if abs(p - target) / target > tol:
        raise ValueError(f"Cannot match capacity within {tol:.0%}: MLP {p} vs SAGE {target}")
    return best_m, p, target
