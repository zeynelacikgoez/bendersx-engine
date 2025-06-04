"""Simplified subproblem worker used in tests."""

from __future__ import annotations

from typing import Tuple

from .config import BendersConfig
from .shared_memory import csr_from_shared


def solve_subproblem_worker(args) -> Tuple[str, float, list, list, list, tuple | None]:
    block_id, start, end, A_meta, B_meta, d, x_prev, r_i_assigned, cfg_dict = args
    _ = csr_from_shared(A_meta)
    _ = csr_from_shared(B_meta)
    n_block = max(0, end - start)
    x_block = [0.0 for _ in range(n_block)]
    return block_id, 0.0, x_block, [], [], None
