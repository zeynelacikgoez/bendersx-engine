from bendersx_engine.matrix_generation import generate_sparse_matrices


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
