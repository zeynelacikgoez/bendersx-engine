from bendersx_engine.benchmark import run_comprehensive_benchmark
from bendersx_engine.benchmark import run_planwirtschaft_benchmark


def test_benchmark_runs():
    run_comprehensive_benchmark()


def test_planwirtschaft_benchmark_runs():
    run_planwirtschaft_benchmark()
