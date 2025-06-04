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
