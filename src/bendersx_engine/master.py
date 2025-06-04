"""Simplified master problem solver used in tests."""

from __future__ import annotations


def solve_master_problem(blocks_metadata, m0, total_r, cuts, config):
    b = len(blocks_metadata)
    r_vars = [[total_r[j] / b for j in range(m0)] for _ in range(b)]
    theta = [0.0 for _ in range(b)]
    return r_vars, theta
