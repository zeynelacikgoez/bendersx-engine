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

New options include hierarchical priorities via `priority_levels`, seasonal
scaling of demand with `seasonal_demand_weights` and ecological costs using
`co2_penalties`. Production cell restrictions and trade limits can be set with
`production_limits` and `import_export_limits`. Configuration files can be
loaded through `PlanwirtschaftParams.from_file` or `BendersConfig.from_file`.
An additional `inventory_cost` can penalize leftover production when
`planwirtschaft_objective` is enabled.
Parallel subproblem solving can be enabled with `use_parallel_subproblems` while
`dynamic_block_weights` updates resource weights between iterations using the
previous distribution.
