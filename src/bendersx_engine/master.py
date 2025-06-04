"""Simplified master problem solver used in tests."""

from __future__ import annotations


def solve_master_problem(blocks_metadata, m0, total_r, cuts, config):
    b = len(blocks_metadata)

    distribution = config.matrix_gen_params.get("block_distribution")
    if distribution and len(distribution) == b:
        weights = [float(w) for w in distribution]
    else:
        weights = [1.0 for _ in range(b)]

    priority = config.matrix_gen_params.get("priority_sectors", [])
    factor = getattr(config, "priority_sector_allocation_factor", 1.0)
    for idx in priority:
        if 0 <= idx < b:
            weights[idx] *= factor

    sum_weights = sum(weights) if weights else 1.0
    r_vars = [
        [total_r[j] * weights[idx] / sum_weights for j in range(m0)] for idx in range(b)
    ]

    theta = [0.0 for _ in range(b)]
    return r_vars, theta
