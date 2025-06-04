# Usage

Run the CLI module to execute the benchmark and show the deployment guide.

The solver stops early when successive iterates change less than the
`convergence_tolerance` defined in `BendersConfig`.

The `partitioning` module splits blocks adaptively based on the dual gap. This
helps keep workloads balanced for large-scale planning models such as
Leontief-style input-output systems.
