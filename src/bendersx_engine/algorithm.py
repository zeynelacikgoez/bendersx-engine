"""Simplified Benders decomposition driver used in tests."""

from __future__ import annotations

from typing import Tuple, List, Dict

from .config import BendersConfig
from .shared_memory import csr_to_shared, cleanup_shared_memory
from .master import solve_master_problem
from .subproblem import solve_subproblem_worker
from .cuts import pareto_select_cuts
from .partitioning import repartition_blocks


def benders_decomposition(
    n: int,
    m0: int,
    total_r,
    A_sparse,
    B_sparse,
    config: BendersConfig | None = None,
    problem_type: str = "general",
) -> Tuple[float, list, List, Dict]:
    if config is None:
        config = BendersConfig()

    cleanup_shared_memory()
    A_meta = csr_to_shared("A", A_sparse)
    B_meta = csr_to_shared("B", B_sparse)

    x_prev = [0.0 for _ in range(n)]
    blocks_metadata = [("block_0", 0, n)]
    all_cuts: List = []

    for _ in range(config.max_iterations_per_phase):
        r_vars, theta = solve_master_problem(
            blocks_metadata, m0, total_r, all_cuts, config
        )
        args_list = []
        for block_id, start, end in blocks_metadata:
            r_i = r_vars[0]
            args_list.append(
                (
                    block_id,
                    start,
                    end,
                    A_meta,
                    B_meta,
                    [0.0] * n,
                    x_prev,
                    r_i,
                    config.__dict__,
                )
            )
        results = [solve_subproblem_worker(args) for args in args_list]
        new_cuts = [res[-1] for res in results if res[-1] is not None]
        all_cuts.extend(new_cuts)
        all_cuts = pareto_select_cuts(all_cuts, config.cut_pool_multiplier, config)
        x_prev = [0.0 for _ in range(n)]
        for block_id, obj, x_block, *_ in results:
            for bid, s, e in blocks_metadata:
                if bid == block_id:
                    x_prev[s:e] = x_block
                    break
        dual_gaps = {
            bid: theta[idx] - obj for idx, (bid, _, _) in enumerate(blocks_metadata)
        }
        blocks_metadata = repartition_blocks(blocks_metadata, dual_gaps, n)

    cleanup_shared_memory()
    total = sum(x_prev)
    return float(total), x_prev, all_cuts, {}
