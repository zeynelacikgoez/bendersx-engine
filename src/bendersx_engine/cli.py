"""Command line interface."""

from .benchmark import run_comprehensive_benchmark
from .deploy import show_deployment_guide


def main() -> None:
    """Entry point for command line."""
    print(__doc__)
    run_comprehensive_benchmark()
    show_deployment_guide()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
