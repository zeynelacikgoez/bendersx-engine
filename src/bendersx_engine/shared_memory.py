"""Shared memory helpers for sparse matrices."""
from __future__ import annotations

import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Dict

import numpy as np
import scipy.sparse as sp

_shared_pool: Dict[str, SharedMemory] = {}


def csr_to_shared(name_prefix: str, csr_matrix: sp.csr_matrix) -> Dict[str, object]:
    """Store CSR matrix in shared memory and return metadata."""
    name_data = f"{name_prefix}_{uuid.uuid4().hex}_data"
    name_indices = f"{name_prefix}_{uuid.uuid4().hex}_indices"
    name_indptr = f"{name_prefix}_{uuid.uuid4().hex}_indptr"

    shm_data = SharedMemory(create=True, size=csr_matrix.data.nbytes, name=name_data)
    shm_indices = SharedMemory(create=True, size=csr_matrix.indices.nbytes, name=name_indices)
    shm_indptr = SharedMemory(create=True, size=csr_matrix.indptr.nbytes, name=name_indptr)

    np.ndarray(csr_matrix.data.shape, dtype=csr_matrix.data.dtype, buffer=shm_data.buf)[:] = csr_matrix.data
    np.ndarray(csr_matrix.indices.shape, dtype=csr_matrix.indices.dtype, buffer=shm_indices.buf)[:] = csr_matrix.indices
    np.ndarray(csr_matrix.indptr.shape, dtype=csr_matrix.indptr.dtype, buffer=shm_indptr.buf)[:] = csr_matrix.indptr

    _shared_pool[name_data] = shm_data
    _shared_pool[name_indices] = shm_indices
    _shared_pool[name_indptr] = shm_indptr

    return {
        "shape": csr_matrix.shape,
        "data_name": name_data,
        "data_dtype": str(csr_matrix.data.dtype),
        "data_shape": csr_matrix.data.shape,
        "indices_name": name_indices,
        "indices_dtype": str(csr_matrix.indices.dtype),
        "indices_shape": csr_matrix.indices.shape,
        "indptr_name": name_indptr,
        "indptr_dtype": str(csr_matrix.indptr.dtype),
        "indptr_shape": csr_matrix.indptr.shape,
    }


def csr_from_shared(meta: Dict[str, object]) -> sp.csr_matrix:
    """Reconstruct CSR matrix from shared memory."""
    shm_d = SharedMemory(name=meta["data_name"])
    shm_i = SharedMemory(name=meta["indices_name"])
    shm_p = SharedMemory(name=meta["indptr_name"])

    data = np.ndarray(meta["data_shape"], dtype=meta["data_dtype"], buffer=shm_d.buf)
    indices = np.ndarray(meta["indices_shape"], dtype=meta["indices_dtype"], buffer=shm_i.buf)
    indptr = np.ndarray(meta["indptr_shape"], dtype=meta["indptr_dtype"], buffer=shm_p.buf)

    return sp.csr_matrix((data, indices, indptr), shape=meta["shape"])


def cleanup_shared_memory() -> None:
    """Release all shared memory segments."""
    for shm in _shared_pool.values():
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    _shared_pool.clear()
