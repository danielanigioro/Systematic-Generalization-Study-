# src/data.py
# Generate datasets and provide a tiny Dataset + collation utilities.

import json, pathlib, random, torch
from torch.utils.data import Dataset
from .config import load_config
from .grammar import sample_command
from .executor import execute

def _write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")

def _mk_items(n, seed_base, max_depth, add_adj=False, min_len=None, max_len=None, tpl_filter=None):
    """Programmatically create n (tokens, template, actions) triples."""
    items = []
    s = seed_base
    depth = max_depth
    attempts = 0
    while len(items) < n:
        toks, tpl = sample_command(s, max_depth=depth, add_adj=add_adj)
        s += 1
        attempts += 1

        if tpl_filter and not tpl_filter(tpl):
            continue
        if min_len and len(toks) < min_len:
            # every 500 rejections, try deeper structures to reach the min length
            if attempts % 500 == 0 and depth < 5:
                depth += 1
            continue
        if max_len and len(toks) > max_len:
            continue

        actions = execute(toks)
        items.append({"tokens": toks, "template": tpl, "actions": actions})

        if len(items) % 500 == 0:
            print(f"[gen] {len(items)}/{n} (depth={depth}, attempts={attempts})")
    return items

# ---------------- Dataset & vocab utilities ----------------

SPECIAL = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}

def build_vocab(seqs, base=None):
    """Build token->id vocab (optionally extending an existing base)."""
    vocab = dict(SPECIAL if base is None else base)
    for s in seqs:
        for tok in s:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    inv = {i: t for t, i in vocab.items()}
    return vocab, inv

def invert_vocab(v: dict) -> dict:
    """id->token map for pretty-printing (used by train.py)."""
    return {i: t for t, i in v.items()}

class CmdActDataset(Dataset):
    def __init__(self, path, tok_vocab=None, act_vocab=None):
        rows = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
        toks = [r["tokens"] for r in rows]
        acts = [r["actions"] for r in rows]
        self.tok_vocab, _ = build_vocab(toks, tok_vocab)
        # include BOS/EOS in the action vocab
        self.act_vocab, _ = build_vocab([["<bos>","<eos>"]] + acts, act_vocab)
        self.data = [(toks[i], acts[i]) for i in range(len(rows))]

    def encode_toks(self, seq):
        return [self.tok_vocab.get(x, SPECIAL["<unk>"]) for x in seq]

    def encode_acts(self, seq):
        ids = [self.act_vocab["<bos>"]] + [self.act_vocab[x] for x in seq] + [self.act_vocab["<eos>"]]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t, a = self.data[idx]
        return (
            torch.tensor(self.encode_toks(t), dtype=torch.long),
            torch.tensor(self.encode_acts(a), dtype=torch.long),
        )

def pad_collate(batch, pad_id=0):
    """Pad variable-length token/action sequences for a batch."""
    import torch.nn.utils.rnn as rnn
    xs, ys = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xs])
    y_lens = torch.tensor([len(y) for y in ys])
    X = rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
    Y = rnn.pad_sequence(ys, batch_first=True, padding_value=pad_id)
    return X, x_lens, Y, y_lens

# ---------------- CLI entry point to generate splits ----------------

def main():
    cfg = load_config()
    out = pathlib.Path(cfg["out_dir"])

    # ----------------------------
    # IID
    # ----------------------------
    p = cfg["splits"]["iid"]
    _write_jsonl(
        out / "iid_train.jsonl",
        _mk_items(
            n=p["n_train"], seed_base=1,
            max_depth=p["max_depth"], add_adj=False,
            min_len=None, max_len=p["max_len"]
        )
    )
    _write_jsonl(
        out / "iid_val.jsonl",
        _mk_items(
            n=p["n_val"], seed_base=10_001,
            max_depth=p["max_depth"], add_adj=False,
            min_len=None, max_len=p["max_len"]
        )
    )
    _write_jsonl(
        out / "iid_test.jsonl",
        _mk_items(
            n=p["n_test"], seed_base=20_001,
            max_depth=p["max_depth"], add_adj=False,
            min_len=None, max_len=p["max_len"]
        )
    )

    # ----------------------------
    # Length generalization
    # ----------------------------
    p = cfg["splits"]["length"]
    # Train/Val: enforce short commands
    _write_jsonl(
        out / "len_train.jsonl",
        _mk_items(
            n=p["n_train"], seed_base=30_001,
            max_depth=2, add_adj=False,
            min_len=None, max_len=p["train_max_len"]
        )
    )
    _write_jsonl(
        out / "len_val.jsonl",
        _mk_items(
            n=p["n_val"], seed_base=40_001,
            max_depth=2, add_adj=False,
            min_len=None, max_len=p["train_max_len"]
        )
    )
    # Test: allow deeper composition to reach longer sequences
    test_depth = p.get("test_max_depth", 3)
    _write_jsonl(
        out / "len_test.jsonl",
        _mk_items(
            n=p["n_test"], seed_base=50_001,
            max_depth=test_depth, add_adj=False,
            min_len=p["test_min_len"], max_len=p["test_max_len"]
        )
    )

    # ----------------------------
    # Template-held-out
    # ----------------------------
    p = cfg["splits"]["template"]
    hold = set(p["holdout_templates"])
    def not_held(tpl): return tpl not in hold
    def is_held(tpl):  return tpl in hold

    _write_jsonl(
        out / "tpl_train.jsonl",
        _mk_items(
            n=p["n_train"], seed_base=60_001,
            max_depth=2, add_adj=False,
            min_len=None, max_len=None, tpl_filter=not_held
        )
    )
    _write_jsonl(
        out / "tpl_val.jsonl",
        _mk_items(
            n=p["n_val"], seed_base=70_001,
            max_depth=2, add_adj=False,
            min_len=None, max_len=None, tpl_filter=not_held
        )
    )
    _write_jsonl(
        out / "tpl_test.jsonl",
        _mk_items(
            n=p["n_test"], seed_base=80_001,
            max_depth=2, add_adj=False,
            min_len=None, max_len=None, tpl_filter=is_held
        )
    )

    # ----------------------------
    # Distractor robustness
    # ----------------------------
    p = cfg["splits"]["distractor"]
    _write_jsonl(
        out / "dist_train.jsonl",
        _mk_items(
            n=p["n_train"], seed_base=90_001,
            max_depth=2, add_adj=True,
            min_len=None, max_len=None
        )
    )
    _write_jsonl(
        out / "dist_val.jsonl",
        _mk_items(
            n=p["n_val"], seed_base=100_001,
            max_depth=2, add_adj=True,
            min_len=None, max_len=None
        )
    )
    _write_jsonl(
        out / "dist_test.jsonl",
        _mk_items(
            n=p["n_test"], seed_base=110_001,
            max_depth=2, add_adj=True,
            min_len=None, max_len=None
        )
    )

if __name__ == "__main__":
    main()
