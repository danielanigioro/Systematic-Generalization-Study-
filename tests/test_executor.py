from src.executor import execute
def test_simple_walk_left_twice():
    acts = execute(["walk","left","twice"])
    assert acts == ["LTURN","WALK","LTURN","WALK"]

