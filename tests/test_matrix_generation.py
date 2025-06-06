from bendersx_engine.matrix_generation import generate_sparse_matrices
from bendersx_engine import BendersConfig, PlanwirtschaftParams


def test_matrix_shapes():
    A, B = generate_sparse_matrices(10, 3)
    assert A.shape[0] == 10
    assert B.shape[0] == 3


def test_planwirtschaft_problem_type():
    A, B = generate_sparse_matrices(5, 2, problem_type="planwirtschaft")
    assert A.shape[0] == 5
    diag_vals = [A.data[i][i] for i in range(5)]
    assert all(v > 0 for v in diag_vals)

    # Column sums should remain below one for input-output models
    for j in range(5):
        col_sum = sum(A.data[i][j] for i in range(5))
        assert col_sum < 0.96

    # Each row of B should have at least one non-zero entry
    for row in B.data:
        assert any(val != 0 for val in row)


def test_planwirtschaft_row_targets_and_limits():
    cfg = BendersConfig(verbose=False, matrix_gen_params={
        "B_row_targets": {0: {1: 0.5}},
        "A_column_limits": {0: 0.1},
    })
    A, B = generate_sparse_matrices(3, 2, problem_type="planwirtschaft", config=cfg)
    assert B.data[0][1] == 0.5
    col_sum = sum(A.data[i][0] for i in range(3))
    assert col_sum <= 0.1 + 1e-9


def test_capacity_limits_and_tech_factor():
    cfg = BendersConfig(verbose=False, matrix_gen_params={
        "priority_sectors": [1],
        "priority_sector_tech_factor": 0.5,
        "sector_capacity_limits": {1: 0.05},
    })
    A, B = generate_sparse_matrices(4, 3, problem_type="planwirtschaft", config=cfg)
    row_sum = sum(B.data[1])
    assert row_sum <= 0.05 + 1e-9
    avg_row0 = sum(A.data[1]) / len(A.data[1])
    avg_other = sum(A.data[0]) / len(A.data[0])
    assert avg_row0 <= avg_other


def test_row_total_targets_dataclass():
    params = PlanwirtschaftParams(B_row_total_targets={0: 2.0})
    cfg = BendersConfig(verbose=False, matrix_gen_params=params)
    A, B = generate_sparse_matrices(3, 1, problem_type="planwirtschaft", config=cfg)
    assert abs(sum(B.data[0]) - 2.0) < 1e-6


def test_priority_levels_and_seasonal_weights():
    params = PlanwirtschaftParams(
        priority_sectors=[0],
        priority_levels={0: 2},
        seasonal_demand_weights=[2.0],
        B_row_targets={0: {0: 0.5}},
    )
    cfg = BendersConfig(verbose=False, matrix_gen_params=params)
    _, B = generate_sparse_matrices(3, 1, problem_type="planwirtschaft", config=cfg)
    assert abs(B.data[0][0] - 1.2) < 1e-6


def test_production_and_trade_limits():
    params = PlanwirtschaftParams(
        production_limits={0: {0: 0.2}},
        import_export_limits={"import": {0: 0.1}, "export": {0: 0.3}},
    )
    cfg = BendersConfig(verbose=False, matrix_gen_params=params)
    A, B = generate_sparse_matrices(2, 1, problem_type="planwirtschaft", config=cfg)
    assert B.data[0][0] <= 0.2 + 1e-9
    col_sum = sum(A.data[i][0] for i in range(2))
    assert col_sum <= 0.1 + 1e-9
    row_sum = sum(B.data[0])
    assert row_sum <= 0.3 + 1e-9
