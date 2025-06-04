import numpy as np
import scipy.sparse as sp
from bendersx_engine.subproblem import solve_subproblem_worker
from bendersx_engine.shared_memory import csr_to_shared, cleanup_shared_memory
from bendersx_engine import BendersConfig


def test_subproblem_worker_runs():
    cfg = BendersConfig(verbose=False)
    A = sp.identity(4, format="csr")
    B = sp.csr_matrix(np.ones((1, 4)))
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args = (
        "b0",
        0,
        4,
        A_meta,
        B_meta,
        np.zeros(4),
        np.zeros(4),
        np.ones(1),
        cfg.__dict__,
    )
    res = solve_subproblem_worker(args)
    cleanup_shared_memory()
    assert res[0] == "b0"
