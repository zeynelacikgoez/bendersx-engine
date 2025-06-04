import numpy as np
import scipy.sparse as sp
from bendersx_engine.master import solve_master_problem
from bendersx_engine import BendersConfig


def test_master_runs():
    cfg = BendersConfig(verbose=False)
    blocks = [("b0", 0, 2)]
    m0 = 1
    total_r = np.array([1.0])
    cuts = []
    solve_master_problem(blocks, m0, total_r, cuts, cfg)


def test_priority_sector_allocation():
    cfg = BendersConfig(
        verbose=False,
        priority_sector_allocation_factor=2.0,
        matrix_gen_params={"priority_sectors": [0]},
    )
    blocks = [("b0", 0, 1), ("b1", 1, 2)]
    m0 = 1
    total_r = np.array([9.0])
    cuts = []
    r_vars, _ = solve_master_problem(blocks, m0, total_r, cuts, cfg)
    assert r_vars[0][0] > r_vars[1][0]
