"""Simplified Benders decomposition driver used in tests."""

from __future__ import annotations

from typing import Tuple, List, Dict
import multiprocessing as mp

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

    iterations_run = 0
    for _ in range(config.max_iterations_per_phase):
        iterations_run += 1
        x_prev_old = x_prev[:]
        r_vars, theta = solve_master_problem(
            blocks_metadata, m0, total_r, all_cuts, config
        )
        args_list = []
        for idx, (block_id, start, end) in enumerate(blocks_metadata):
            r_i = r_vars[idx]
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
        if config.use_parallel_subproblems:
            with mp.Pool(processes=config.n_processes) as pool:
                results = pool.map(solve_subproblem_worker, args_list)
        else:
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
        if config.dynamic_block_weights:
            new_dist = []
            prev_dist = config.matrix_gen_params.get("block_distribution") or [1.0 for _ in blocks_metadata]
            for idx, (_, start, end) in enumerate(blocks_metadata):
                planned = sum(r_vars[idx])
                produced = sum(x_prev[start:end])
                ratio = produced / planned if planned > 0 else 1.0
                weight = 0.5 * prev_dist[idx] + 0.5 * ratio
                new_dist.append(weight)
            config.matrix_gen_params["block_distribution"] = new_dist
        dual_gaps = {
            bid: theta[idx] - results[idx][1]
            for idx, (bid, _, _) in enumerate(blocks_metadata)
        }
        blocks_metadata = repartition_blocks(blocks_metadata, dual_gaps, n)

        diff = sum(abs(x_prev[i] - x_prev_old[i]) for i in range(n))
        if diff < config.convergence_tolerance:
            break

    cleanup_shared_memory()
    total = sum(x_prev)
    unfulfilled = []
    for idx, (_, start, end) in enumerate(blocks_metadata):
        produced = sum(x_prev[start:end])
        planned = sum(r_vars[idx])
        unfulfilled.append(max(0.0, planned - produced))

    return float(total), x_prev, all_cuts, {
        "iterations": iterations_run,
        "unfulfilled_demand": unfulfilled,
    }
