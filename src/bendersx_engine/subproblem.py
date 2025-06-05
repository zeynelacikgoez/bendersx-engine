"""Simplified subproblem worker used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence, List, Iterable

from .config import BendersConfig
from .shared_memory import csr_from_shared
from .cuts import make_opt_cut


@dataclass
class SubproblemInput:
    block_id: str
    start: int
    end: int
    A_meta: dict
    B_meta: dict
    d: Sequence[float]
    x_prev: Sequence[float]
    r_i_assigned: Sequence[float]
    config: BendersConfig


@dataclass
class SubproblemOutput:
    block_id: str
    obj: float
    x_block: list
    pi_i: list
    mu_iT_d_value: float
    cut: tuple | None


def _apply_tiered_penalty(dev: float, tiers: Iterable[tuple]) -> float:
    """Compute penalty for a deviation using tiered (piecewise) coefficients."""
    remaining = dev
    prev = 0.0
    total = 0.0
    tiers_list = list(tiers)
    for thresh, coeff in tiers_list:
        if remaining <= 0:
            break
        step = min(remaining, thresh - prev)
        if step > 0:
            total += step * coeff
            remaining -= step
        prev = thresh
    if remaining > 0 and tiers_list:
        total += remaining * tiers_list[-1][1]
    return total


def solve_subproblem_worker(args) -> Tuple[str, float, list, list, list, tuple | None]:
    if isinstance(args, SubproblemInput):
        inp = args
    else:
        block_id, start, end, A_meta, B_meta, d, x_prev, r_i_assigned, cfg_dict = args
        cfg = BendersConfig(**cfg_dict)
        inp = SubproblemInput(block_id, start, end, A_meta, B_meta, d, x_prev, r_i_assigned, cfg)

    A = csr_from_shared(inp.A_meta)
    B = csr_from_shared(inp.B_meta)
    n_block = max(0, inp.end - inp.start)

    demand = sum(
        sum(B.data[i][inp.start:inp.end]) * inp.r_i_assigned[i]
        for i in range(len(inp.r_i_assigned))
    )
    x_block = [demand / n_block if n_block > 0 else 0.0 for _ in range(n_block)]

    obj = sum(x_block)
    if inp.config.matrix_gen_params.get("planwirtschaft_objective"):
        under_penalty = inp.config.matrix_gen_params.get("underproduction_penalty", 1.0)
        over_penalty = inp.config.matrix_gen_params.get("overproduction_penalty", 0.0)
        under_penalties = inp.config.matrix_gen_params.get("underproduction_penalties")
        over_penalties = inp.config.matrix_gen_params.get("overproduction_penalties")
        tiered_under = inp.config.matrix_gen_params.get("tiered_underproduction_penalties")
        tiered_over = inp.config.matrix_gen_params.get("tiered_overproduction_penalties")
        prod_bonus = inp.config.matrix_gen_params.get("production_bonus", 0.0)
        bonus_factor = inp.config.matrix_gen_params.get("priority_sector_bonus_factor", 1.0)
        priority = set(inp.config.matrix_gen_params.get("priority_sectors", []))

        m0 = len(inp.r_i_assigned)
        if under_penalties is None:
            under_penalties = [under_penalty for _ in range(m0)]
        if over_penalties is None:
            over_penalties = [over_penalty for _ in range(m0)]

        produced_vec = []
        for i in range(m0):
            prod = 0.0
            for j in range(n_block):
                prod += B.data[i][inp.start + j] * x_block[j]
            produced_vec.append(prod)

        obj = 0.0
        for i in range(m0):
            planned = inp.r_i_assigned[i]
            produced = produced_vec[i]
            under_dev = max(0.0, planned - produced)
            over_dev = max(0.0, produced - planned)
            if tiered_under:
                under_cost = _apply_tiered_penalty(under_dev, tiered_under)
            else:
                under_cost = under_penalties[i] * under_dev
            if tiered_over:
                over_cost = _apply_tiered_penalty(over_dev, tiered_over)
            else:
                over_cost = over_penalties[i] * over_dev
            sector_bonus = prod_bonus
            if i in priority:
                sector_bonus *= bonus_factor
            bonus = sector_bonus * min(planned, produced)
            obj += produced + bonus - under_cost - over_cost
    pi_i = [0.5 for _ in inp.r_i_assigned]
    mu_iT_d_value = obj - sum(pi_i[j] * inp.r_i_assigned[j] for j in range(len(pi_i)))
    cut = make_opt_cut(inp.block_id, pi_i, mu_iT_d_value)
    return inp.block_id, obj, x_block, pi_i, [mu_iT_d_value], cut
