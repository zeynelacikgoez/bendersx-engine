from bendersx_engine.matrix_generation import generate_sparse_matrices
from bendersx_engine import BendersConfig


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
