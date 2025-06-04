from bendersx_engine.env_detection import NUMBA_AVAILABLE


def test_numba_flag():
    assert isinstance(NUMBA_AVAILABLE, bool)
