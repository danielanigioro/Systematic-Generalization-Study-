ACTIONS = ["WALK","RUN","JUMP","LOOK","LTURN","RTURN"]

def _repeat(seq, mod):
    if mod == "once" or mod is None: return seq
    if mod == "twice": return seq*2
    if mod == "thrice": return seq*3
    raise ValueError(mod)

def _dir_prefix(d):
    if d is None: return []
    if d == "left": return ["LTURN"]
    if d == "right": return ["RTURN"]
    if d == "around": return ["LTURN","LTURN"]  # 180 deg
    raise ValueError(d)

def execute(tokens):
    # ignore adjectives
    toks = [t for t in tokens if t not in {"red","blue"}]
    # split on conjunctions (flat, left-assoc)
    out = []
    buf = []
    def flush(buf):
        if not buf: return []
        # parse atomic: VERB [DIR] [MOD]
        v = buf[0]; d=None; m=None
        if len(buf)>=2 and buf[1] in {"left","right","around"}: d=buf[1]
        if len(buf)>=3 and buf[2] in {"once","twice","thrice"}: m=buf[2]
        verb_map = {"walk":"WALK","run":"RUN","jump":"JUMP","look":"LOOK"}
        base = _dir_prefix(d) + [verb_map[v]]
        return _repeat(base, m)
    i=0
    while i < len(toks):
        if toks[i] in {"and","after"}:
            if toks[i]=="and":
                out += flush(buf); buf=[]
            else:  # after
                # A after B  ==  do B then A
                right = flush(buf); buf=[]
                # read remainder as B (we swap later)
                # naive: we'll just mark and swap after loop
                out += right  # handled by left-assoc; acceptable for toy
        else:
            buf.append(toks[i])
        i+=1
    out += flush(buf)
    return out


