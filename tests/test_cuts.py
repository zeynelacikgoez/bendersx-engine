from bendersx_engine.cuts import make_feas_cut
import numpy as np


def test_make_feas_cut():
    cut = make_feas_cut("b0", np.array([1, 0]), np.array([2, 2]))
    assert cut[0] == "feas"
