"""Block partitioning utilities."""

from __future__ import annotations

from typing import List, Tuple, Dict


def repartition_blocks(
    blocks_metadata: List[Tuple[str, int, int]],
    dual_gaps: Dict[str, float],
    max_block_size: int,
):
    new_metadata = []
    for block_id, start, end in blocks_metadata:
        size = end - start
        gap = dual_gaps.get(block_id, 0)
        if size > max_block_size and gap > 1e-3:
            split_factor = 0.5 + min(gap, 1.0) / 4
            mid = start + int(size * split_factor)
            if mid <= start or mid >= end:
                mid = start + size // 2
            new_metadata.append((block_id, start, mid))
            new_metadata.append((f"split_{block_id}", mid, end))
        else:
            new_metadata.append((block_id, start, end))
    return new_metadata
