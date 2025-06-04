from __future__ import annotations

import random

from .simple_matrix import SimpleMatrix
from .optimizations import sparse_norm_optimized
from .config import BendersConfig
from .env_detection import NUMBA_AVAILABLE


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

        # Ensure column sums remain below one as in classical
        # input-output models. Scale columns exceeding the threshold.
        for j in range(n):
            col_sum = sum(A.data[i][j] for i in range(n))
            if col_sum >= 0.95:
                factor = 0.95 / col_sum
                for i in range(n):
                    A.data[i][j] *= factor

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
    else:
        B = _random_matrix(m0, n, sparsity * 2)

    return A, B
