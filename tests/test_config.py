from bendersx_engine import BendersConfig


def test_config_defaults():
    cfg = BendersConfig()
    assert cfg.highs_threads > 0
