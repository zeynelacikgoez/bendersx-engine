"""Adaptive Big-M calculation."""

from __future__ import annotations

import math
from .config import BendersConfig


def _inf_norm(values):
    return max(abs(v) for v in values) if values else 0.0


def calculate_adaptive_big_m(c_sub, b_ub_core, config: BendersConfig) -> float:
    c_norm = _inf_norm(c_sub) if len(c_sub) > 0 else 1.0
    b_norm = _inf_norm(b_ub_core) if len(b_ub_core) > 0 else 1.0
    scale = config.adaptive_big_m_factor
    big_m = scale * max(c_norm, b_norm, 1.0)
    return min(big_m, config.big_m_cap)
