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


def test_tiered_penalties():
    cfg_linear = BendersConfig(
        verbose=False,
        matrix_gen_params={"planwirtschaft_objective": True, "underproduction_penalty": 1.0},
    )
    cfg_tiered = BendersConfig(
        verbose=False,
        matrix_gen_params={
            "planwirtschaft_objective": True,
            "tiered_underproduction_penalties": [(1.0, 1.0), (2.0, 2.0)],
        },
    )
    A = sp.identity(1, format="csr")
    B = sp.csr_matrix([[0.5]])
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_lin = (
        "b0",
        0,
        1,
        A_meta,
        B_meta,
        np.zeros(1),
        np.zeros(1),
        np.array([3.0]),
        cfg_linear.__dict__,
    )
    _, obj_lin, *_ = solve_subproblem_worker(args_lin)
    cleanup_shared_memory()
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_tier = (
        "b0",
        0,
        1,
        A_meta,
        B_meta,
        np.zeros(1),
        np.zeros(1),
        np.array([3.0]),
        cfg_tiered.__dict__,
    )
    _, obj_tier, *_ = solve_subproblem_worker(args_tier)
    cleanup_shared_memory()
    assert obj_tier < obj_lin


def test_production_bonus():
    cfg_base = BendersConfig(
        verbose=False,
        matrix_gen_params={"planwirtschaft_objective": True},
    )
    cfg_bonus = BendersConfig(
        verbose=False,
        matrix_gen_params={"planwirtschaft_objective": True, "production_bonus": 0.5},
    )
    A = sp.identity(1, format="csr")
    B = sp.csr_matrix([[2.0]])
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_base = (
        "b0",
        0,
        1,
        A_meta,
        B_meta,
        np.zeros(1),
        np.zeros(1),
        np.array([1.0]),
        cfg_base.__dict__,
    )
    _, obj_base, *_ = solve_subproblem_worker(args_base)
    cleanup_shared_memory()

    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_bonus = (
        "b0",
        0,
        1,
        A_meta,
        B_meta,
        np.zeros(1),
        np.zeros(1),
        np.array([1.0]),
        cfg_bonus.__dict__,
    )
    _, obj_bonus, *_ = solve_subproblem_worker(args_bonus)
    cleanup_shared_memory()

    assert obj_bonus > obj_base


def test_societal_bonus():
    cfg_base = BendersConfig(verbose=False, matrix_gen_params={"planwirtschaft_objective": True})
    cfg_bonus = BendersConfig(verbose=False, matrix_gen_params={"planwirtschaft_objective": True, "societal_bonuses": {0: 0.5}})
    A = sp.identity(1, format="csr")
    B = sp.csr_matrix([[2.0]])
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_base = ("b0", 0, 1, A_meta, B_meta, np.zeros(1), np.zeros(1), np.array([1.0]), cfg_base.__dict__)
    _, obj_base, *_ = solve_subproblem_worker(args_base)
    cleanup_shared_memory()

    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_bonus = ("b0", 0, 1, A_meta, B_meta, np.zeros(1), np.zeros(1), np.array([1.0]), cfg_bonus.__dict__)
    _, obj_bonus, *_ = solve_subproblem_worker(args_bonus)
    cleanup_shared_memory()

    assert obj_bonus > obj_base


def test_co2_penalty():
    cfg_pen = BendersConfig(verbose=False, matrix_gen_params={"planwirtschaft_objective": True, "co2_penalties": {0: 1.0}})
    A = sp.identity(1, format="csr")
    B = sp.csr_matrix([[2.0]])
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args = ("b0", 0, 1, A_meta, B_meta, np.zeros(1), np.zeros(1), np.array([1.0]), cfg_pen.__dict__)
    _, obj_pen, *_ = solve_subproblem_worker(args)
    cleanup_shared_memory()
    assert obj_pen < 1.0


def test_inventory_cost():
    cfg_base = BendersConfig(verbose=False, matrix_gen_params={"planwirtschaft_objective": True})
    cfg_inv = BendersConfig(verbose=False, matrix_gen_params={"planwirtschaft_objective": True, "inventory_cost": 0.5})
    A = sp.identity(1, format="csr")
    B = sp.csr_matrix([[2.0]])
    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_base = ("b0", 0, 1, A_meta, B_meta, np.zeros(1), np.zeros(1), np.array([0.5]), cfg_base.__dict__)
    _, obj_base, *_ = solve_subproblem_worker(args_base)
    cleanup_shared_memory()

    A_meta = csr_to_shared("A", A)
    B_meta = csr_to_shared("B", B)
    args_inv = ("b0", 0, 1, A_meta, B_meta, np.zeros(1), np.zeros(1), np.array([0.5]), cfg_inv.__dict__)
    _, obj_inv, *_ = solve_subproblem_worker(args_inv)
    cleanup_shared_memory()

    assert obj_inv < obj_base
