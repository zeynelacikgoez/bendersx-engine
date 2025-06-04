"""Lightweight performance stubs used for tests."""

from __future__ import annotations

from .simple_matrix import SimpleMatrix


def create_identity_optimized(n: int, format: str = "csr", dtype=float):
    """Return a simple identity matrix."""
    data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return SimpleMatrix(data)


def sparse_matvec_optimized(data, indices, indptr, x):
    """Minimal sparse matrix-vector product used in stubs."""
    result = [0.0 for _ in range(len(indptr) - 1)]
    idx = 0
    for i in range(len(indptr) - 1):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        result[i] = s
    return result


def sparse_norm_optimized(data):
    return max(abs(v) for v in data) if data else 0.0


def setup_highs_optimized(highs_instance, config):
    """Dummy configuration helper for HiGHS."""
    return False
