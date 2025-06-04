import numpy as np
from bendersx_engine.bigm import calculate_adaptive_big_m
from bendersx_engine import BendersConfig


def test_big_m_positive():
    cfg = BendersConfig()
    val = calculate_adaptive_big_m(np.array([1, 2]), np.array([3]), cfg)
    assert val > 0
