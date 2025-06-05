import numpy as np
import scipy.sparse as sp
from bendersx_engine.algorithm import benders_decomposition
from bendersx_engine import BendersConfig


def test_algorithm_runs():
    cfg = BendersConfig(verbose=False)
    n, m0 = 4, 1
    A = sp.identity(n, format="csr")
    B = sp.csr_matrix(np.ones((m0, n)))
    total_r = np.ones(m0)
    _, _, _, info = benders_decomposition(n, m0, total_r, A, B, cfg)
    assert info["iterations"] <= cfg.max_iterations_per_phase
    assert isinstance(info.get("unfulfilled_demand"), list)
