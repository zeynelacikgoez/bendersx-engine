# BendersX Engine

This repository contains a small Python implementation demonstrating the core steps of Benders decomposition. The code is intentionally minimal and serves mainly as a teaching and testing aid.

## Features

* **Benders Driver** – a simplified `benders_decomposition` routine coordinating master and subproblem calls.
* **Configurable Solver** – basic configuration via `BendersConfig` with options such as thread counts and JIT usage.
* **Stub Components** – lightweight master and subproblem solvers and utilities for cuts, block partitioning and shared-memory storage.
* **Environment Helpers** – checks for optional packages like HiGHS and Numba.
* **Examples and Tests** – a CLI entry point (`python -m bendersx_engine.cli`) and a small pytest suite.
* **Early Stopping** – iterations halt once the solution change drops below a configurable tolerance.
* **Adaptive Partitioning** – block sizes adjust according to the dual gap to better handle large planning models.
* **Planned Economy Support** – generate matrices for `planwirtschaft` style input-output systems using the existing helpers.
* **Hierarchical Priorities** – optional `priority_levels` and seasonal weights allow refined demand modeling with ecological penalties.
* **Production and Trade Limits** – additional parameters bound individual production cells and import/export totals.
* **Config Loading** – `PlanwirtschaftParams.from_file` and `BendersConfig.from_file` allow JSON based configuration files.

The implementation is not intended for production use. It omits many optimizations and contains simplified placeholders.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/zeynelacikgoez/bendersx-engine.git
cd bendersx-engine
pip install -e .
```

## Running the Example

Execute the command line interface to generate matrices, run the driver and print a short benchmark:

```bash
python -m bendersx_engine.cli
```

The module `scripts/run_benders.py` provides a small wrapper for this command.

## Development

Run the test suite with:

```bash
pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
