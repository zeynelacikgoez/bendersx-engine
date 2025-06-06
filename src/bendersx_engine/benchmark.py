"""Benchmark helpers."""

from __future__ import annotations

from .config import BendersConfig
from .matrix_generation import generate_sparse_matrices
from .algorithm import benders_decomposition


def run_comprehensive_benchmark() -> None:
    config = BendersConfig(verbose=False)
    n, m0 = 100, 10
    total_r = [1.0 for _ in range(m0)]
    A, B = generate_sparse_matrices(n, m0, 0.01, "leontief", config)
    obj, _, cuts, _ = benders_decomposition(n, m0, total_r, A, B, config, "leontief")
    print(f"Objective: {obj:.2f}, cuts: {len(cuts)}")


def run_planwirtschaft_benchmark() -> None:
    config = BendersConfig(
        verbose=False,
        matrix_gen_params={"planwirtschaft_objective": True, "priority_sectors": [0]},
    )
    n, m0 = 20, 5
    total_r = [1.0 for _ in range(m0)]
    A, B = generate_sparse_matrices(n, m0, 0.05, "planwirtschaft", config)
    obj, _, cuts, _ = benders_decomposition(
        n, m0, total_r, A, B, config, "planwirtschaft"
    )
    print(f"Planwirtschaft objective: {obj:.2f}, cuts: {len(cuts)}")
