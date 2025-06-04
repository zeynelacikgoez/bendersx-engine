import numpy as np
from bendersx_engine.optimizations import create_identity_optimized


def test_identity():
    I = create_identity_optimized(3)
    assert I.shape == (3, 3)
