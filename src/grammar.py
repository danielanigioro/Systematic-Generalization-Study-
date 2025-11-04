import random
VERBS = ["walk","run","jump","look"]
DIRS  = ["left","right","around"]
MODS  = ["once","twice","thrice"]
CONJ  = ["and","after"]
ADJS  = ["red","blue"]  # ignored by executor (distractors)

def _maybe_adj(rng, add_adj): return [rng.choice(ADJS)] if add_adj and rng.random()<0.5 else []

def atomic_cmd(rng, add_adj=False):
    v = rng.choice(VERBS)
    d = rng.choice(DIRS) if rng.random()<0.7 else None
    m = rng.choice(MODS) if rng.random()<0.5 else None
    toks = _maybe_adj(rng, add_adj) + [v] + ([d] if d else []) + ([m] if m else [])
    tpl = "V" + ("-D" if d else "") + ("-M" if m else "")
    return toks, tpl

def compose(rng, max_depth=2, add_adj=False):
    if max_depth<=1 or rng.random()<0.6:
        return atomic_cmd(rng, add_adj)
    left, lt = compose(rng, max_depth-1, add_adj)
    right, rt = compose(rng, max_depth-1, add_adj)
    conj = rng.choice(CONJ)
    toks = left + [conj] + right
    tpl = f"({lt}){conj.upper()}({rt})"
    return toks, tpl

def sample_command(seed, max_depth=2, add_adj=False):
    rng = random.Random(seed)
    return compose(rng, max_depth=max_depth, add_adj=add_adj)
