"""Simplified subproblem worker used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence

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
    pi_i = [0.5 for _ in inp.r_i_assigned]
    mu_iT_d_value = obj - sum(pi_i[j] * inp.r_i_assigned[j] for j in range(len(pi_i)))
    cut = make_opt_cut(inp.block_id, pi_i, mu_iT_d_value)
    return inp.block_id, obj, x_block, pi_i, [mu_iT_d_value], cut
