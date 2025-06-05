import json
from bendersx_engine import BendersConfig, PlanwirtschaftParams


def test_config_defaults():
    cfg = BendersConfig()
    assert cfg.highs_threads > 0


def test_config_from_file(tmp_path):
    data = {
        "verbose": False,
        "matrix_gen_params": {"diag_base": 0.3},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))
    cfg = BendersConfig.from_file(str(path))
    assert cfg.verbose is False
    assert cfg.matrix_gen_params["diag_base"] == 0.3


def test_planwirtschaft_params_from_file(tmp_path):
    data = {"diag_base": 0.4, "diag_variation": 0.5}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(data))
    params = PlanwirtschaftParams.from_file(str(path))
    assert params.diag_base == 0.4
    assert params.diag_variation == 0.5
