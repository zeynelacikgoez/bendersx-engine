"""Master problem solver."""
from __future__ import annotations

import numpy as np
import highspy

from .optimizations import setup_highs_optimized

last_basis = None
persistent_highs_master = None


def solve_master_problem(blocks_metadata, m0, total_r, cuts, config):
    global last_basis, persistent_highs_master

    b = len(blocks_metadata)
    num_r = b * m0
    num_theta = b

    if persistent_highs_master is None:
        persistent_highs_master = highspy.Highs()
    else:
        try:
            persistent_highs_master.clear()
        except Exception:
            persistent_highs_master = highspy.Highs()

    setup_highs_optimized(persistent_highs_master, config)

    col_costs = np.concatenate([np.zeros(num_r), -np.ones(num_theta)])
    col_lower = np.concatenate([np.zeros(num_r), np.full(num_theta, -np.inf)])
    col_upper = np.concatenate([np.full(num_r, np.inf), np.full(num_theta, np.inf)])
    persistent_highs_master.addCols(num_r + num_theta, col_lower, col_upper, col_costs)

    for j in range(m0):
        row_indices = [i * m0 + j for i in range(b)]
        row_coeffs = [1.0] * b
        persistent_highs_master.addRow(total_r[j], total_r[j], row_indices, row_coeffs)

    block_pos = {bid: idx for idx, (bid, _, _) in enumerate(blocks_metadata)}

    for cut in cuts:
        ctype, block_id, beta, alpha = cut
        if block_id not in block_pos:
            continue
        idx = block_pos[block_id]
        row_indices = []
        row_coeffs = []
        if ctype == "opt":
            row_indices.append(num_r + idx)
            row_coeffs.append(1.0)
            for j in range(min(m0, len(beta))):
                row_indices.append(idx * m0 + j)
                row_coeffs.append(-beta[j])
            persistent_highs_master.addRow(-highspy.kHighsInf, alpha, row_indices, row_coeffs)
        else:
            for j in range(min(m0, len(beta))):
                row_indices.append(idx * m0 + j)
                row_coeffs.append(beta[j])
            persistent_highs_master.addRow(alpha, highspy.kHighsInf, row_indices, row_coeffs)

    if last_basis is not None:
        try:
            persistent_highs_master.setBasis(last_basis)
        except Exception:
            last_basis = None

    persistent_highs_master.run()
    status = persistent_highs_master.getModelStatus()

    if status == highspy.HighsModelStatus.kOptimal:
        sol = persistent_highs_master.getSolution().col_value
        last_basis = persistent_highs_master.getBasis()
        r_flat = sol[:num_r]
        theta = sol[num_r:]
        return r_flat.reshape((b, m0)), theta

    r_vars = np.tile(total_r / b, (b, 1))
    theta = np.full(b, -np.inf)
    last_basis = None
    return r_vars, theta
