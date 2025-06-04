"""Performance optimizations using Numba and HiGHS."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .env_detection import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True)
    def sparse_matvec_optimized(data, indices, indptr, x):
        result = np.zeros(indptr.size - 1)
        for i in prange(indptr.size - 1):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * x[indices[j]]
            result[i] = s
        return result

    @njit(parallel=True, fastmath=True, cache=True)
    def sparse_norm_optimized(data):
        m = 0.0
        for i in prange(len(data)):
            v = abs(data[i])
            if v > m:
                m = v
        return m
else:

    def sparse_matvec_optimized(data, indices, indptr, x):  # pragma: no cover
        csr = sp.csr_matrix((data, indices, indptr))
        return csr.dot(x)

    def sparse_norm_optimized(data):  # pragma: no cover
        return np.max(np.abs(data))


def create_identity_optimized(n: int, format: str = "csr", dtype=np.float64):
    return sp.identity(n, format=format, dtype=dtype)


def setup_highs_optimized(highs_instance, config):
    try:
        highs_instance.setOptionValue("time_limit", config.highs_time_limit)
        if config.use_highs_threading:
            highs_instance.setOptionValue("threads", config.highs_threads)
        if config.use_first_order_gpu:
            highs_instance.setOptionValue("solver", "firstorder")
        highs_instance.setOptionValue("presolve", "on")
        highs_instance.setOptionValue("parallel", "on")
        if not config.verbose:
            highs_instance.setOptionValue("output_flag", False)
        return True
    except Exception:
        return False
