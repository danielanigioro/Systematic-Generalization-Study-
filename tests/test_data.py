import json, pathlib
def count(path): return sum(1 for _ in open(path))
def test_has_files():
    d = pathlib.Path("data/generated")
    assert (d/"iid_train.jsonl").exists()
    assert count(d/"iid_train.jsonl") > 100

