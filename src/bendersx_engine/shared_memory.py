"""Simplified shared memory helpers for tests."""

from __future__ import annotations

from typing import Dict, Any, cast
from .simple_matrix import SimpleMatrix

_shared_store: Dict[str, SimpleMatrix] = {}


def csr_to_shared(name_prefix: str, csr_matrix: SimpleMatrix) -> Dict[str, object]:
    _shared_store[name_prefix] = csr_matrix
    return {"name": name_prefix}


def csr_from_shared(meta: Dict[str, Any]) -> SimpleMatrix:
    return cast(SimpleMatrix, _shared_store.get(meta["name"]))


def cleanup_shared_memory() -> None:
    _shared_store.clear()
