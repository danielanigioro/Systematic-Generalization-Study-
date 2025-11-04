# src/aggregate_human.py
import csv, argparse, os
from collections import defaultdict

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # split tokens/gold/human back to lists
            row["tokens"] = row["tokens"].split()
            row["gold"]   = row["gold"].split()
            row["human"]  = row["human"].split()
            row["correct"] = int(row.get("correct", int(row["gold"] == row["human"])))
            yield row

def agg(rows, key_fn):
    ok = defaultdict(int); tot = defaultdict(int)
    for row in rows:
        k = key_fn(row)
        ok[k]  += row["correct"]
        tot[k] += 1
    out = []
    for k in ok:
        acc = ok[k] / max(1, tot[k])
        out.append((k, ok[k], tot[k], acc))
    return sorted(out, key=lambda x: x[0])

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_csv", default="results/human.csv")
    ap.add_argument("--out_len",   default="results/human_acc_by_length.csv")
    ap.add_argument("--out_tpl",   default="results/human_acc_by_template.csv")
    args = ap.parse_args()

    rows = list(read_rows(args.human_csv))

    # accuracy by input length (tokens)
    by_len = agg(rows, key_fn=lambda r: len(r["tokens"]))
    write_csv(args.out_len, ["length","ok","tot","acc"], by_len)
    print(f"[saved] {args.out_len} ({len(by_len)} buckets)")

    # accuracy by template string
    by_tpl = agg(rows, key_fn=lambda r: r.get("template","?"))
    write_csv(args.out_tpl, ["template","ok","tot","acc"], by_tpl)
    print(f"[saved] {args.out_tpl} ({len(by_tpl)} templates)")

if __name__ == "__main__":
    main()

