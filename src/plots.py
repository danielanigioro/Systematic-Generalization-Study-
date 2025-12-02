# src/plots.py
import argparse, csv, os
import matplotlib.pyplot as plt

SPLIT_ORDER = ["iid", "distractor", "template", "length"]
DISPLAY = {
    "iid": "IID",
    "distractor": "Distractor",
    "template": "Template-held-out",
    "length": "Length generalization",
}

# ---------------------------
# Overall bar (from results.csv)
# ---------------------------
def plot_overall(results_csv, out_png):
    rows = []
    with open(results_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append((row["split"], float(row["exact"])))
            except Exception:
                pass
    latest = {}
    for s, acc in rows:
        latest[s] = acc
    splits = [s for s in SPLIT_ORDER if s in latest]
    accs = [latest[s] for s in splits]
    labels = [DISPLAY.get(s, s) for s in splits]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1.05)
    plt.ylabel("Exact-match accuracy")
    plt.title("Overall accuracy by split")
    for i, a in enumerate(accs):
        plt.text(i, min(1.02, a + 0.02), f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")

# ---------------------------
# Accuracy vs input length
# ---------------------------
def plot_acc_by_length(acc_by_length_csv, out_png, len_min=None, len_max=None):
    L, A = [], []
    with open(acc_by_length_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            L.append(int(row["length"]))
            A.append(float(row["acc"]))
    z = sorted(zip(L, A), key=lambda x: x[0])
    if len_min is not None or len_max is not None:
        lo = len_min if len_min is not None else -1e9
        hi = len_max if len_max is not None else 1e9
        z = [p for p in z if lo <= p[0] <= hi]
    L = [x for x, _ in z]
    A = [y for _, y in z]

    plt.figure(figsize=(6, 4))
    plt.plot(L, A, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Input length (tokens)")
    plt.ylabel("Exact-match accuracy")
    plt.title("Length generalization: accuracy vs. length")
    for x, y in z:
        plt.text(x, min(1.02, y + 0.03), f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")

# ---------------------------
# Accuracy per template
# ---------------------------
def plot_acc_by_template(acc_by_template_csv, out_png, top_k=None):
    TPL, A = [], []
    with open(acc_by_template_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            TPL.append(row["template"])
            A.append(float(row["acc"]))
    # sort by accuracy ascending to highlight the hard ones
    z = sorted(zip(TPL, A), key=lambda x: x[1])
    if top_k:
        z = z[:top_k]
    labels = [t for t, _ in z]
    accs = [a for _, a in z]

    plt.figure(figsize=(max(6, 0.4 * len(labels)), 4))
    plt.bar(range(len(labels)), accs)
    plt.ylim(0, 1.05)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Exact-match accuracy")
    plt.title("Template-held-out: per-template accuracy")
    for i, a in enumerate(accs):
        plt.text(i, min(1.02, a + 0.02), f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")

# ---------------------------
# Human vs Model overlays
# ---------------------------
def _read_len_csv(path):
    L, A = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            L.append(int(row["length"]))
            A.append(float(row["acc"]))
    z = sorted(zip(L, A), key=lambda x: x[0])
    return [x for x, _ in z], [y for _, y in z]

def overlay_length(model_csv, human_csv, out_png):
    Lm, Am = _read_len_csv(model_csv)
    Lh, Ah = _read_len_csv(human_csv)

    plt.figure(figsize=(6, 4))
    plt.plot(Lm, Am, marker="o", label="Model")
    plt.plot(Lh, Ah, marker="s", label="Human")
    plt.ylim(0, 1.05)
    plt.xlabel("Input length (tokens)")
    plt.ylabel("Exact-match accuracy")
    plt.title("Accuracy vs length (Human vs Model)")
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")

def _read_tpl_csv(path):
    T, A = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            T.append(row["template"])
            A.append(float(row["acc"]))
    return T, A

def overlay_template(model_csv, human_csv, out_png, top_k=None):
    Tm, Am = _read_tpl_csv(model_csv)
    Th, Ah = _read_tpl_csv(human_csv)

    # align by templates present in either (union), then sort by model accuracy asc
    tpl_set = set(Tm) | set(Th)
    md = {t: a for t, a in zip(Tm, Am)}
    hd = {t: a for t, a in zip(Th, Ah)}
    pairs = []
    for t in tpl_set:
        pairs.append((t, md.get(t, 0.0), hd.get(t, 0.0)))
    pairs.sort(key=lambda x: x[1])  # by model acc

    if top_k:
        pairs = pairs[:top_k]

    labels = [t for t, _, _ in pairs]
    model_acc = [a for _, a, _ in pairs]
    human_acc = [a for _, _, a in pairs]

    ix = range(len(labels))
    plt.figure(figsize=(max(6, 0.5 * len(labels)), 4))
    plt.bar(ix, model_acc, label="Model", alpha=0.7)
    plt.bar(ix, human_acc, label="Human", alpha=0.5)
    plt.ylim(0, 1.05)
    plt.xticks(ix, labels, rotation=45, ha="right")
    plt.ylabel("Exact-match accuracy")
    plt.title("Per-template accuracy (Human vs Model)")
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # existing
    ap.add_argument("--overall", type=str, help="Path to results/results.csv")
    ap.add_argument("--out_overall", type=str, default="results/overall_acc.png")
    ap.add_argument("--by_length", type=str, help="Path to results/acc_by_length.csv")
    ap.add_argument("--out_by_length", type=str, default="results/acc_by_length.png")
    ap.add_argument("--len_min", type=int, default=None, help="Optional minimum length to display")
    ap.add_argument("--len_max", type=int, default=None, help="Optional maximum length to display")
    ap.add_argument("--by_template", type=str, help="Path to results/acc_by_template.csv")
    ap.add_argument("--out_by_template", type=str, default="results/acc_by_template.png")
    ap.add_argument("--top_k", type=int, default=None, help="Show only the hardest K templates")

    # new overlay flags
    ap.add_argument("--overlay_len_model", type=str, help="Model acc_by_length.csv")
    ap.add_argument("--overlay_len_human", type=str, help="Human human_acc_by_length.csv")
    ap.add_argument("--out_overlay_len", type=str, default="results/overlay_len_human_model.png")

    ap.add_argument("--overlay_tpl_model", type=str, help="Model acc_by_template.csv")
    ap.add_argument("--overlay_tpl_human", type=str, help="Human human_acc_by_template.csv")
    ap.add_argument("--out_overlay_tpl", type=str, default="results/overlay_tpl_human_model.png")

    args = ap.parse_args()

    if args.overall:
        plot_overall(args.overall, args.out_overall)
    if args.by_length:
        plot_acc_by_length(args.by_length, args.out_by_length, args.len_min, args.len_max)
    if args.by_template:
        plot_acc_by_template(args.by_template, args.out_by_template, top_k=args.top_k)

    if args.overlay_len_model and args.overlay_len_human:
        overlay_length(args.overlay_len_model, args.overlay_len_human, args.out_overlay_len)
    if args.overlay_tpl_model and args.overlay_tpl_human:
        overlay_template(args.overlay_tpl_model, args.overlay_tpl_human, args.out_overlay_tpl, top_k=args.top_k)

if __name__ == "__main__":
    main()
