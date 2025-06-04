"""Matrix generation utilities."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .optimizations import sparse_norm_optimized
from .config import BendersConfig
from .env_detection import NUMBA_AVAILABLE


def generate_sparse_matrices(n: int, m0: int, sparsity: float = 0.02, problem_type: str = "general", config: BendersConfig | None = None):
    if config is None:
        config = BendersConfig()

    if problem_type == "leontief":
        A = sp.random(n, n, density=sparsity, format="csr", dtype=np.float64)
        A.setdiag(0.1 + 0.8 * np.random.random(n))
    else:
        A = sp.random(n, n, density=sparsity, format="csr", dtype=np.float64)

    if config.use_numba_jit and NUMBA_AVAILABLE:
        rho = sparse_norm_optimized(A.data)
    else:
        rho = sp.linalg.norm(A, ord=np.inf)

    if rho > 0:
        A.data *= 0.95 / rho

    B = sp.random(m0, n, density=sparsity * 2, format="csr", dtype=np.float64)
    return A, B
