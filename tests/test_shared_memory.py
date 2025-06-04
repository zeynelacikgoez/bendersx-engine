import numpy as np
import scipy.sparse as sp
from bendersx_engine.shared_memory import (
    csr_to_shared,
    csr_from_shared,
    cleanup_shared_memory,
)


def test_shared_roundtrip():
    A = sp.identity(4, format="csr")
    meta = csr_to_shared("test", A)
    B = csr_from_shared(meta)
    cleanup_shared_memory()
    assert (A != B).nnz == 0
