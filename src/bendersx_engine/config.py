"""Solver configuration."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .env_detection import check_highs_version, detect_gpu_support, setup_numba_cache


@dataclass
class PlanwirtschaftParams:
    """Structured parameters for planwirtschaft-style matrix generation."""

    diag_base: float = 0.2
    diag_variation: float = 0.7
    max_col_sum_A: float = 0.95
    A_column_limits: Dict[int, float] | None = None
    B_row_targets: Dict[int, Dict[int, float]] | None = None
    B_row_total_targets: Dict[int, float] | None = None
    priority_sectors: List[int] = field(default_factory=list)
    priority_sector_demand_factor: float = 1.0
    priority_sector_tech_factor: float | None = None
    sector_capacity_limits: Dict[int, float] | None = None
    underproduction_penalty: float = 1.0
    overproduction_penalty: float = 0.0
    underproduction_penalties: List[float] | None = None
    overproduction_penalties: List[float] | None = None
    planwirtschaft_objective: bool = False
    tiered_underproduction_penalties: List[tuple] | None = None
    tiered_overproduction_penalties: List[tuple] | None = None
    production_bonus: float = 0.0
    priority_sector_bonus_factor: float = 1.0
    societal_bonuses: Dict[int, float] | None = None
    min_block_allocations: Dict[int, float] | None = None
    priority_levels: Dict[int, int] | None = None
    seasonal_demand_weights: List[float] | None = None
    co2_penalties: Dict[int, float] | None = None
    production_limits: Dict[int, Dict[int, float]] | None = None
    import_export_limits: Dict[int, Dict[str, float]] | None = None

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_file(path: str) -> "PlanwirtschaftParams":
        """Load parameters from a JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PlanwirtschaftParams(**data)


@dataclass
class BendersConfig:
    """Configuration for BendersX Engine."""

    n_processes: Optional[int] = None
    verbose: bool = True
    use_highs_threading: bool = True
    highs_threads: int = 8
    highs_time_limit: float = 300.0
    use_numba_jit: bool = True
    use_identity_fast: bool = True
    use_first_order_gpu: bool = False
    max_iterations_per_phase: int = 3
    cut_pool_multiplier: int = 8
    adaptive_big_m_factor: int = 50
    big_m_cap: float = 1e7
    convergence_tolerance: float = 1e-6
    matrix_gen_params: Optional[dict | PlanwirtschaftParams] = None  # e.g. planwirtschaft parameters
    enable_memory_tracking: bool = True
    set_omp_threads: bool = True
    priority_sector_allocation_factor: float = 1.0
    use_parallel_subproblems: bool = False
    dynamic_block_weights: bool = False

    def __post_init__(self) -> None:
        if self.n_processes is None:
            self.n_processes = min(8, mp.cpu_count())

        if self.set_omp_threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(self.highs_threads))
            if self.verbose:
                print(f"OpenMP threads: {self.highs_threads}")

        setup_numba_cache()

        if self.use_first_order_gpu:
            if not detect_gpu_support():
                self.use_first_order_gpu = False

        if self.use_highs_threading:
            check_highs_version()

        if self.matrix_gen_params is None:
            self.matrix_gen_params = {}
        elif isinstance(self.matrix_gen_params, PlanwirtschaftParams):
            self.matrix_gen_params = self.matrix_gen_params.to_dict()

        if self.matrix_gen_params.get("priority_sectors") and self.priority_sector_allocation_factor < 1.0:
            raise ValueError("priority_sector_allocation_factor must be >= 1.0")

    @staticmethod
    def from_file(path: str) -> "BendersConfig":
        """Load configuration from a JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("matrix_gen_params")
        if isinstance(params, dict):
            data["matrix_gen_params"] = PlanwirtschaftParams(**params)
        return BendersConfig(**data)
