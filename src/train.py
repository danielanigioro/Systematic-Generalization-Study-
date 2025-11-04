# Parse config, build dataset/dataloader, build model from src/models/*, train with AdamW,
# save best checkpoint and append a CSV row to results/results.csv.

import os, yaml, random, numpy as np, torch, csv
from torch.utils.data import DataLoader
from src.data import CmdActDataset, pad_collate, SPECIAL, invert_vocab

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def accuracy_exact(logits, gold, pad_id=0, eos_id=2):
    # greedy exact-match on full sequence (up to eos)
    preds = logits.argmax(-1)  # (B,Tpred)
    B = preds.size(0)
    correct = 0
    for i in range(B):
        # trim at first eos (or end)
        def trim(seq):
            out = []
            for t in seq.tolist():
                if t == pad_id:
                    continue
                if t == eos_id:
                    break
                out.append(t)
            return out
        if trim(preds[i]) == trim(gold[i, 1:]):  # gold excludes initial <bos>
            correct += 1
    return correct, B

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    # ----- robust type-casts from YAML -----
    set_seed(int(cfg.get("seed", 2400)))

    data_dir = cfg.get("data_dir", "data/generated")
    split = cfg.get("split", "iid")
    train_file = cfg.get("train_file", f"{split}_train.jsonl")
    val_file   = cfg.get("val_file",   f"{split}_val.jsonl")
    test_file  = cfg.get("test_file",  f"{split}_test.jsonl")

    embed_dim     = int(cfg.get("embed_dim", 128))
    hidden_dim    = int(cfg.get("hidden_dim", 256))
    num_layers    = int(cfg.get("num_layers", 2))
    dropout       = float(cfg.get("dropout", 0.1))

    batch_size    = int(cfg.get("batch_size", 128))
    lr            = float(cfg.get("lr", 3e-4))
    epochs        = int(cfg.get("epochs", 10))
    teacher_force = float(cfg.get("teacher_force", 0.5))
    save_best     = bool(cfg.get("save_best", True))
    model_name    = str(cfg.get("model", "lstm_attn")).lower()

    # ----- datasets -----
    train_ds = CmdActDataset(os.path.join(data_dir, train_file))
    val_ds   = CmdActDataset(os.path.join(data_dir, val_file),  train_ds.tok_vocab, train_ds.act_vocab)
    test_ds  = CmdActDataset(os.path.join(data_dir, test_file), train_ds.tok_vocab, train_ds.act_vocab)

    tok_vocab, act_vocab = train_ds.tok_vocab, train_ds.act_vocab

    # ----- model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "lstm_attn":
        from src.models.lstm_attn import Seq2SeqLSTM
        model = Seq2SeqLSTM(tok_vocab, act_vocab,
                            emb=embed_dim, hid=hidden_dim,
                            layers=num_layers, dropout=dropout).to(device)
    elif model_name == "transformer":
        from src.models.transformer import Seq2SeqTransformer
        d_model = embed_dim
        nhead   = int(cfg.get("nhead", 4))
        dim_ff  = int(cfg.get("ff_dim", 512))
        layers  = num_layers
        model = Seq2SeqTransformer(tok_vocab, act_vocab,
                                   d_model=d_model, nhead=nhead,
                                   num_layers=layers, dim_ff=dim_ff,
                                   dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL["<pad>"])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=pad_collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    best_acc = -1.0
    save_dir = cfg.get("out_dir", "results")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{model_name}_{split}.pt")

    # ---------------- training ----------------
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0
        steps = 0
        for X, Xl, Y, Yl in train_dl:
            X, Xl, Y = X.to(device), Xl.to(device), Y.to(device)
            T = Y.size(1)  # includes <bos> ... <eos>
            logits = model(X, Xl, y=Y, tf=teacher_force, max_steps=T)  # (B,T?,|V|)
            # Align logits (T-1 steps) to targets Y[:,1:] (drop <bos>)
            logits = logits[:, :T-1, :]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           Y[:, 1:].contiguous().reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); steps += 1

        # quick val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Xl, Y, Yl in val_dl:
                X, Xl, Y = X.to(device), Xl.to(device), Y.to(device)
                T = Y.size(1)
                logits = model(X, Xl, y=None, tf=0.0, max_steps=T)  # greedy decode for eval
                c, b = accuracy_exact(logits, Y)
                correct += c; total += b
        acc = correct / total if total else 0.0
        print(f"[ep {ep}] train_loss={tot_loss/max(1,steps):.4f}  val_exact={acc:.3f}")
        if save_best and acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(),
                        "tok_vocab": tok_vocab, "act_vocab": act_vocab}, ckpt_path)
            print(f"  saved → {ckpt_path}")

    # ---------------- final test ----------------
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    model.eval(); correct = total = 0

    # invert vocabs for readable examples
    tok_inv = invert_vocab(tok_vocab)
    act_inv = invert_vocab(act_vocab)

    # collect a few example predictions for the README/slides
    examples = []

    with torch.no_grad():
        for X, Xl, Y, Yl in test_dl:
            X, Xl, Y = X.to(device), Xl.to(device), Y.to(device)
            T = Y.size(1)
            logits = model(X, Xl, y=None, tf=0.0, max_steps=T)
            c, b = accuracy_exact(logits, Y); correct += c; total += b

            # capture a few example (ids) triples
            pred = logits.argmax(-1)
            for i in range(min(5 - len(examples), X.size(0))):
                examples.append((X[i].tolist(), Y[i].tolist(), pred[i].tolist()))
                if len(examples) >= 5:
                    break

    print(f"[test {split}] exact-match = {correct}/{total} = {correct/total:.3f}")
    print("[examples] show 3 preds vs gold (IDs):")
    for xtoks, ygold, ypred in examples[:3]:
        print("  x:", xtoks)
        print("  y:", ygold)
        print("  p:", ypred)

    # pretty-printed tokens/actions
    print("[examples] pretty tokens/actions:")
    for xtoks, ygold, ypred in examples[:3]:
        # input tokens (skip <pad>=0)
        xt = [tok_inv.get(i, "?") for i in xtoks if i != 0]
        # gold actions (drop <bos>=1, stop at <eos>=2, skip <pad>=0)
        yg = []
        for i in ygold[1:]:
            if i in (0, 2):  # pad or eos
                if i == 2: break
                continue
            yg.append(act_inv.get(i, "?"))
        # predicted actions (stop at eos, skip pad)
        yp = []
        for i in ypred:
            if i in (0, 2):
                if i == 2: break
                continue
            yp.append(act_inv.get(i, "?"))

        print("  x:", xt)
        print("  y:", yg)
        print("  p:", yp)

    # ---------------- CSV logging ----------------
    csv_path = os.path.join(save_dir, "results.csv")
    header = ["split", "epochs", "batch_size", "lr", "exact_raw", "exact"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([split, epochs, batch_size, lr, f"{correct}/{total}", f"{correct/total:.3f}"])

if __name__ == "__main__":
    main()
