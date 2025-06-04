from __future__ import annotations

from typing import List

from bendersx_engine.simple_matrix import SimpleMatrix


class csr_matrix(SimpleMatrix):
    pass


def identity(n: int, format: str = "csr", dtype=float) -> csr_matrix:
    data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return csr_matrix(data)


def vstack(mats: List[csr_matrix], format: str = "csr") -> csr_matrix:
    data = []
    for m in mats:
        data.extend([row[:] for row in m.data])
    return csr_matrix(data)


def hstack(mats: List[csr_matrix], format: str = "csr") -> csr_matrix:
    if not mats:
        return csr_matrix([])
    rows = mats[0].shape[0]
    data = []
    for r in range(rows):
        row = []
        for m in mats:
            row.extend(m.data[r])
        data.append(row)
    return csr_matrix(data)
