"""Subproblem solver."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .bigm import calculate_adaptive_big_m
from .config import BendersConfig
from .shared_memory import csr_from_shared
from .optimizations import create_identity_optimized, sparse_matvec_optimized


def solve_subproblem_worker(args) -> Tuple[str, float, np.ndarray, np.ndarray, np.ndarray, tuple | None]:
    block_id, start, end, A_meta, B_meta, d, x_prev, r_i_assigned, cfg_dict = args
    config = BendersConfig(**cfg_dict)

    A_data = csr_from_shared(A_meta)
    B_data = csr_from_shared(B_meta)

    n_total = A_data.shape[1]
    n_block = end - start

    if n_block <= 0:
        return block_id, 0.0, np.zeros(1), np.zeros(1), np.zeros(1), None

    A_block = A_data[start:end, start:end]
    B_block = B_data[:, start:end]

    cross = np.zeros(n_block)
    if start > 0:
        cross -= sparse_matvec_optimized(A_data[start:end, :start].data,
                                         A_data[start:end, :start].indices,
                                         A_data[start:end, :start].indptr,
                                         x_prev[:start])
    if end < n_total:
        cross -= sparse_matvec_optimized(A_data[start:end, end:].data,
                                         A_data[start:end, end:].indices,
                                         A_data[start:end, end:].indptr,
                                         x_prev[end:])

    RHS = d[start:end] - cross
    c_sub = -np.ones(n_block)
    I_block = create_identity_optimized(n_block)
    G1 = -(I_block - A_block)
    h1 = -RHS
    G2 = B_block
    h2 = r_i_assigned

    A_ub_core = sp.vstack([G1, G2], format="csr")
    b_ub_core = np.concatenate([h1, h2])
    big_m = calculate_adaptive_big_m(c_sub, b_ub_core, config)
    num_constraints = A_ub_core.shape[0]
    S_slack = create_identity_optimized(num_constraints, format="csc")
    A_ub = sp.hstack([A_ub_core, -S_slack], format="csr")
    c_elastic = np.concatenate([c_sub, big_m * np.ones(num_constraints)])

    res = linprog(c_elastic, A_ub=A_ub, b_ub=b_ub_core, bounds=[(0, None)] * (n_block + num_constraints), method="highs")

    if res.success:
        x_block = res.x[:n_block]
        return block_id, np.sum(x_block), x_block, np.zeros(G1.shape[0]), np.zeros(G2.shape[0]), None

    return block_id, 0.0, np.zeros(n_block), np.zeros(G1.shape[0]), np.zeros(G2.shape[0]), None
