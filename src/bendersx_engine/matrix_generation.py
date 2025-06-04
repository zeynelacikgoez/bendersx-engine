from __future__ import annotations

import random

from .simple_matrix import SimpleMatrix
from .optimizations import sparse_norm_optimized
from .config import BendersConfig
from .env_detection import NUMBA_AVAILABLE


def _normalize_column_sums(matrix: SimpleMatrix, max_sum: float) -> None:
    for j in range(matrix.shape[1]):
        col_sum = sum(matrix.data[i][j] for i in range(matrix.shape[0]))
        if col_sum >= max_sum:
            factor = max_sum / col_sum
            for i in range(matrix.shape[0]):
                matrix.data[i][j] *= factor


def _ensure_b_rows_nonzero(B: SimpleMatrix) -> None:
    n = B.shape[1]
    for row in B.data:
        if not any(val != 0 for val in row) and n > 0:
            row[random.randrange(n)] = random.random()


def _apply_planwirtschaft_modifiers(A: SimpleMatrix, B: SimpleMatrix, params: dict) -> None:
    diag_vals = [0.2 + 0.7 * random.random() for _ in range(A.shape[0])]
    A.setdiag(diag_vals)
    max_col_sum = params.get("max_col_sum_A", 0.95)
    _normalize_column_sums(A, max_col_sum)
    _ensure_b_rows_nonzero(B)


def _random_matrix(rows: int, cols: int, density: float) -> SimpleMatrix:
    data = []
    for _ in range(rows):
        row = [
            random.random() if random.random() < density else 0.0 for _ in range(cols)
        ]
        data.append(row)
    return SimpleMatrix(data)


def generate_sparse_matrices(
    n: int,
    m0: int,
    sparsity: float = 0.02,
    problem_type: str = "general",
    config: BendersConfig | None = None,
):
    if config is None:
        config = BendersConfig()

    A = _random_matrix(n, n, sparsity)
    if problem_type in {"leontief", "planwirtschaft"}:
        diag_vals = [0.2 + 0.7 * random.random() for _ in range(n)]
        A.setdiag(diag_vals)
        _normalize_column_sums(A, 0.95)

    rho = sparse_norm_optimized([v for row in A.data for v in row])
    if rho > 0:
        scale = 0.95 / rho
        for i in range(n):
            for j in range(n):
                A.data[i][j] *= scale

    if problem_type == "planwirtschaft":
        data = []
        num_entries = max(1, int(n * sparsity))
        for _ in range(m0):
            row = [0.0] * n
            cols = random.sample(range(n), num_entries)
            for j in cols:
                row[j] = random.random()
            data.append(row)
        B = SimpleMatrix(data)
        _apply_planwirtschaft_modifiers(A, B, config.matrix_gen_params)
    else:
        B = _random_matrix(m0, n, sparsity * 2)

    return A, B
