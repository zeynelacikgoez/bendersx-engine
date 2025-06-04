from bendersx_engine.partitioning import repartition_blocks


def test_partitioning_simple():
    blocks = [("b", 0, 10)]
    new = repartition_blocks(blocks, {"b": 2.0}, 5)
    assert len(new) == 2
