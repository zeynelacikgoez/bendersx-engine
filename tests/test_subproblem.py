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


def test_overproduction_penalty():
    cfg = BendersConfig(
        verbose=False,
        matrix_gen_params={"planwirtschaft_objective": True, "overproduction_penalty": 0.5},
    )
    A = sp.identity(2, format="csr")
    B = sp.csr_matrix(np.ones((1, 2)))
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args = (
        "b0",
        0,
        2,
        A_meta,
        B_meta,
        np.zeros(2),
        np.zeros(2),
        np.ones(1),
        cfg.__dict__,
    )
    block_id, obj, *_ = solve_subproblem_worker(args)
    cleanup_shared_memory()
    assert block_id == "b0"
    assert obj < 2.0


def test_componentwise_penalties():
    cfg = BendersConfig(
        verbose=False,
        matrix_gen_params={
            "planwirtschaft_objective": True,
            "underproduction_penalties": [2.0, 1.0],
        },
    )
    A = sp.identity(2, format="csr")
    B = sp.identity(2, format="csr")
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args = (
        "b0",
        0,
        2,
        A_meta,
        B_meta,
        np.zeros(2),
        np.zeros(2),
        np.array([2.0, 0.0]),
        cfg.__dict__,
    )
    _, obj, *_ = solve_subproblem_worker(args)
    cleanup_shared_memory()
    assert obj < 1.5
