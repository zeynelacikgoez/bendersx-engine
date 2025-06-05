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


def _ensure_b_rows_nonzero(
    B: SimpleMatrix,
    row_targets: dict | None = None,
    row_total_targets: dict | None = None,
) -> None:
    """Ensure every row of ``B`` has at least one non-zero entry and optional totals.

    Parameters
    ----------
    B : SimpleMatrix
        Demand matrix to modify in-place.
    row_targets : dict | None, optional
        Optional dictionary mapping row indices to ``{col: value}`` mappings.
        Specified entries are written before the non-zero check is performed.
    row_total_targets : dict | None, optional
        Optional mapping of row indices to desired total row sums. The row will
        be scaled to match this target after ``row_targets`` are applied and
        non-zero entries ensured.
    """
    if row_targets is None:
        row_targets = {}
    if row_total_targets is None:
        row_total_targets = {}

    n = B.shape[1]
    for idx, row in enumerate(B.data):
        target = row_targets.get(idx)
        if target:
            for j, val in target.items():
                if 0 <= j < n:
                    row[j] = val

        if not any(val != 0 for val in row) and n > 0:
            row[random.randrange(n)] = random.random()

        total_target = row_total_targets.get(idx)
        if total_target is not None and n > 0:
            current = sum(row)
            if current > 0:
                factor = total_target / current
                for j in range(n):
                    row[j] *= factor
            else:
                share = total_target / n
                for j in range(n):
                    row[j] = share


def _apply_planwirtschaft_modifiers(A: SimpleMatrix, B: SimpleMatrix, params: dict) -> None:
    """Apply simple structured tweaks for planwirtschaft matrices.

    The helper normalizes ``A`` according to optional column limits and ensures
    the ``B`` matrix adheres to given demand targets.
    """
    diag_base = params.get("diag_base", 0.2)
    diag_var = params.get("diag_variation", 0.7)
    diag_vals = [diag_base + diag_var * random.random() for _ in range(A.shape[0])]
    A.setdiag(diag_vals)

    max_col_sum = params.get("max_col_sum_A", 0.95)

    column_limits = params.get("A_column_limits")
    if isinstance(column_limits, dict):
        for j, limit in column_limits.items():
            if 0 <= j < A.shape[1]:
                col_sum = sum(A.data[i][j] for i in range(A.shape[0]))
                if col_sum > limit and limit > 0:
                    factor = limit / col_sum
                    for i in range(A.shape[0]):
                        A.data[i][j] *= factor

    _normalize_column_sums(A, max_col_sum)

    row_targets = params.get("B_row_targets")
    row_total_targets = params.get("B_row_total_targets")
    _ensure_b_rows_nonzero(B, row_targets, row_total_targets)

    priority_factor = params.get("priority_sector_demand_factor", 1.0)
    priority_sectors = params.get("priority_sectors", [])
    for idx in priority_sectors:
        if 0 <= idx < B.shape[0]:
            row = B.data[idx]
            if not any(val != 0 for val in row) and B.shape[1] > 0:
                row[random.randrange(B.shape[1])] = random.random() * priority_factor
            B.data[idx] = [val * priority_factor for val in row]

    tech_factor = params.get("priority_sector_tech_factor")
    if tech_factor is not None:
        for idx in priority_sectors:
            if 0 <= idx < A.shape[0]:
                A.data[idx] = [val * tech_factor for val in A.data[idx]]

    capacity_limits = params.get("sector_capacity_limits")
    if isinstance(capacity_limits, dict):
        for idx, limit in capacity_limits.items():
            if 0 <= idx < B.shape[0] and limit > 0:
                row_sum = sum(B.data[idx])
                if row_sum > limit:
                    factor = limit / row_sum
                    B.data[idx] = [val * factor for val in B.data[idx]]


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
    if problem_type == "leontief":
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
