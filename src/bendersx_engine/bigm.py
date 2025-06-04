"""Adaptive Big-M calculation."""
from __future__ import annotations

import numpy as np

from .config import BendersConfig


def calculate_adaptive_big_m(c_sub, b_ub_core, config: BendersConfig) -> float:
    c_norm = np.linalg.norm(c_sub, ord=np.inf) if len(c_sub) > 0 else 1.0
    b_norm = np.linalg.norm(b_ub_core, ord=np.inf) if len(b_ub_core) > 0 else 1.0
    scale = config.adaptive_big_m_factor
    big_m = scale * max(c_norm, b_norm, 1.0)
    return min(big_m, config.big_m_cap)
