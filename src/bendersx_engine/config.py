"""Solver configuration."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Optional

from .env_detection import check_highs_version, detect_gpu_support, setup_numba_cache


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
    matrix_gen_params: Optional[dict] = None  # e.g. planwirtschaft parameters
    enable_memory_tracking: bool = True
    set_omp_threads: bool = True
    priority_sector_allocation_factor: float = 1.0

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

        if self.matrix_gen_params.get("priority_sectors") and self.priority_sector_allocation_factor < 1.0:
            raise ValueError("priority_sector_allocation_factor must be >= 1.0")
