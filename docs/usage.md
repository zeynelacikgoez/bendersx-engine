# Usage

Run the CLI module to execute the benchmark and show the deployment guide.

The solver stops early when successive iterates change less than the
`convergence_tolerance` defined in `BendersConfig`.

The `partitioning` module splits blocks adaptively based on the dual gap. This
helps keep workloads balanced for large-scale planning models such as
Leontief-style input-output systems. The same routine can be used for
"planwirtschaft" problems by specifying `problem_type="planwirtschaft"` when
generating matrices. In this mode the matrix generator now normalizes column
sums to stay below one and ensures the demand matrix contains at least one
non-zero entry per row. Additional parameters allow emphasizing
`priority_sectors` in both resource allocation and technology. Set
`priority_sector_allocation_factor` in ``BendersConfig`` to allocate more
resources to these blocks. Component-wise penalties can be configured with
`underproduction_penalties` and `overproduction_penalties` in
``matrix_gen_params``.
