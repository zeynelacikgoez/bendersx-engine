from bendersx_engine.matrix_generation import generate_sparse_matrices


def test_matrix_shapes():
    A, B = generate_sparse_matrices(10, 3)
    assert A.shape[0] == 10
    assert B.shape[0] == 3
