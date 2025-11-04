# src/evaluate.py
import os, json, argparse, torch, csv
from torch.utils.data import DataLoader, Subset
from src.data import CmdActDataset, pad_collate, SPECIAL
from src.models.lstm_attn import Seq2SeqLSTM

def load_ckpt(path, device):
    state = torch.load(path, map_location=device)
    return state["model"], state["tok_vocab"], state["act_vocab"]

def accuracy_exact(logits, gold, pad_id=0, eos_id=2):
    preds = logits.argmax(-1)
    def trim(seq):
        out=[]
        for t in seq.tolist():
            if t==pad_id: continue
            if t==eos_id: break
            out.append(t)
        return out
    B = preds.size(0); ok=0
    for i in range(B):
        if trim(preds[i]) == trim(gold[i,1:]): ok += 1
    return ok, B

def eval_split(ckpt, data_dir, split, test_file, batch=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_ds = CmdActDataset(os.path.join(data_dir, test_file))
    model_state, tok_vocab, act_vocab = load_ckpt(ckpt, device)
    ds = CmdActDataset(os.path.join(data_dir, test_file), tok_vocab, act_vocab)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=pad_collate)
    model = Seq2SeqLSTM(tok_vocab, act_vocab).to(device)
    model.load_state_dict(model_state); model.eval()
    all_ok=all_tot=0
    with torch.no_grad():
        for X,Xl,Y,Yl in dl:
            X,Xl,Y = X.to(device), Xl.to(device), Y.to(device)
            T = Y.size(1)
            logits = model(X, Xl, y=None, tf=0.0, max_steps=T)
            ok,tot = accuracy_exact(logits, Y)
            all_ok += ok; all_tot += tot
    return all_ok, all_tot

def eval_by_length(ckpt, data_path, out_csv):
    rows = [json.loads(x) for x in open(data_path, "r", encoding="utf-8")]
    lengths = [len(r["tokens"]) for r in rows]
    idx_by_L = {}
    for i,L in enumerate(lengths): idx_by_L.setdefault(L, []).append(i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = CmdActDataset(data_path)
    state, tok_vocab, act_vocab = load_ckpt(ckpt, device)
    model = Seq2SeqLSTM(tok_vocab, act_vocab).to(device)
    model.load_state_dict(state); model.eval()
    full = CmdActDataset(data_path, tok_vocab, act_vocab)

    def subset(indices): return Subset(full, indices)

    results = []
    with torch.no_grad():
        for L in sorted(idx_by_L.keys()):
            dl = DataLoader(subset(idx_by_L[L]), batch_size=256, shuffle=False, collate_fn=pad_collate)
            ok=tot=0
            for X,Xl,Y,Yl in dl:
                X,Xl,Y = X.to(device), Xl.to(device), Y.to(device)
                T = Y.size(1)
                logits = model(X, Xl, y=None, tf=0.0, max_steps=T)
                c,b = accuracy_exact(logits, Y); ok+=c; tot+=b
            results.append((L, ok, tot, ok/max(1,tot)))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["length","ok","tot","acc"])
        for r in results: w.writerow(r)

def eval_by_template(ckpt, data_path, out_csv):
    """Groups test examples by their 'template' string and reports per-template exact-match."""
    rows = [json.loads(x) for x in open(data_path, "r", encoding="utf-8")]
    idx_by_tpl = {}
    for i,r in enumerate(rows):
        idx_by_tpl.setdefault(r.get("template","?"), []).append(i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = CmdActDataset(data_path)
    state, tok_vocab, act_vocab = load_ckpt(ckpt, device)
    model = Seq2SeqLSTM(tok_vocab, act_vocab).to(device)
    model.load_state_dict(state); model.eval()
    full = CmdActDataset(data_path, tok_vocab, act_vocab)

    def subset(indices): return Subset(full, indices)

    results = []
    with torch.no_grad():
        for tpl in sorted(idx_by_tpl.keys()):
            dl = DataLoader(subset(idx_by_tpl[tpl]), batch_size=256, shuffle=False, collate_fn=pad_collate)
            ok=tot=0
            for X,Xl,Y,Yl in dl:
                X,Xl,Y = X.to(device), Xl.to(device), Y.to(device)
                T = Y.size(1)
                logits = model(X, Xl, y=None, tf=0.0, max_steps=T)
                c,b = accuracy_exact(logits, Y); ok+=c; tot+=b
            results.append((tpl, ok, tot, ok/max(1,tot)))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["template","ok","tot","acc"])
        for r in results: w.writerow(r)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["iid","length","template","distractor"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/generated")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--by_length_csv", default=None)
    ap.add_argument("--by_template_csv", default=None)
    args = ap.parse_args()

    ok, tot = eval_split(args.ckpt, args.data_dir, args.split, args.test_file)
    print(f"[eval {args.split}] exact = {ok}/{tot} = {ok/max(1,tot):.3f}")

    if args.by_length_csv:
        eval_by_length(args.ckpt, os.path.join(args.data_dir, args.test_file), args.by_length_csv)
        print(f"[saved] {args.by_length_csv}")

    if args.by_template_csv:
        eval_by_template(args.ckpt, os.path.join(args.data_dir, args.test_file), args.by_template_csv)
        print(f"[saved] {args.by_template_csv}")
