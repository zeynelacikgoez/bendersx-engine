from __future__ import annotations

import math
from .config import BendersConfig


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def make_feas_cut(block_id: str, pi_feas_only, r_i_assigned):
    alpha = _dot(pi_feas_only, r_i_assigned)
    beta = list(pi_feas_only)
    return ("feas", block_id, beta, alpha)


def make_opt_cut(block_id: str, pi_i, mu_iT_d_value):
    if any(math.isnan(x) for x in pi_i) or math.isnan(mu_iT_d_value):
        return None
    return ("opt", block_id, list(pi_i), mu_iT_d_value)


def pareto_select_cuts(cuts_list: list, max_k: int, config: BendersConfig):
    if len(cuts_list) <= max_k:
        return cuts_list

    feas_cuts = [c for c in cuts_list if c[0] == "feas"]
    opt_cuts = [c for c in cuts_list if c[0] == "opt"]

    feas_cuts.sort(key=lambda c: abs(c[3]), reverse=True)
    opt_cuts.sort(key=lambda c: abs(c[3]), reverse=True)

    max_feas = max(1, max_k // 3)
    max_opt = max_k - max_feas

    return feas_cuts[:max_feas] + opt_cuts[:max_opt]
