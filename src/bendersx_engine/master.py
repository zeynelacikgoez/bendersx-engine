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

    min_levels = config.matrix_gen_params.get("min_block_allocations", {})
    for j in range(m0):
        min_reqs = [min_levels.get(i, 0.0) * total_r[j] for i in range(b)]
        sum_min = sum(min_reqs)
        if sum_min >= total_r[j]:
            for i in range(b):
                share = min_reqs[i] / sum_min if sum_min > 0 else 1.0 / b
                r_vars[i][j] = total_r[j] * share
            continue

        for i in range(b):
            if min_reqs[i] > r_vars[i][j]:
                r_vars[i][j] = min_reqs[i]
        current = sum(r_vars[i][j] for i in range(b))
        if current > total_r[j]:
            excess = current - total_r[j]
            adjustable = [i for i in range(b) if r_vars[i][j] > min_reqs[i]]
            total_adj = sum(r_vars[i][j] - min_reqs[i] for i in adjustable)
            for i in adjustable:
                reducible = r_vars[i][j] - min_reqs[i]
                reduction = excess * (reducible / total_adj) if total_adj else 0
                r_vars[i][j] -= reduction
        else:
            remaining = total_r[j] - current
            weight_sum = sum(weights)
            for i in range(b):
                r_vars[i][j] += remaining * (weights[i] / weight_sum if weight_sum else 1.0 / b)

    theta = [0.0 for _ in range(b)]
    return r_vars, theta
