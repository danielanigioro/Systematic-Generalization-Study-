
<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/414000d1-0a6e-4547-a275-529be671dbb0" />
<img width="1800" height="1000" alt="overlay_tpl_human_model_demo" src="https://github.com/user-attachments/assets/9e08c52e-d433-4f6d-8938-ed660bda97d4" />
<img width="1600" height="1000" alt="overlay_len_human_model_demo" src="https://github.com/user-attachments/assets/d1731891-3068-4f5f-be0d-e621e564705d" />


# Systematic-Generalization-Study-
This repo evaluates whether small neural networks can follow longer, recombined instructions as reliably as humans. We generate a compact command language (e.g., ‘walk left twice and jump’), map it to action sequences, and train three model families (BiLSTM+Attention, Transformer, and a structure-biased Transformer with an operator bottleneck). We probe systematic compositional generalization with length and template-held-out splits, add distractors to test robustness, collect a human baseline, and report accuracy, calibration (ECE), and the inflection point where models break while humans stay strong. Everything is reproducible via run.sh and documented with plots and a short technical write-up.

# BreakPointAI — Commands→Actions Generalization
**Goal:** Show a clear capability gap: an LSTM nails in-distribution (IID) commands but fails to generalize to longer compositions and held-out templates; humans don’t.

## What’s in this repo
- **Data generator:** programmatically builds splits (IID / Length / Template / Distractor)
- **Baselines:** LSTM seq2seq (+ optional Transformer hook)
- **Human baseline:** Streamlit UI + aggregator
- **Evaluation & plots:** accuracy by length/template, human vs model overlays
- **Repro scripts:** single-command runs on Windows PowerShell

## Quick Start (Windows / PowerShell)
```powershell
# 1) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps (lean)
pip install -r requirements.txt

# 3) Generate data
python -m src.data --config configs\base.yaml

# 4) Train LSTM on a split (edit yaml split/train/val/test files)
python -m src.train --config configs\lstm.yaml

# 5) (Optional) Evaluate + write CSVs (if you added evaluate step)
python -m src.evaluate --config configs\lstm.yaml

# 6) Human baseline UI
streamlit run src\human_app.py
# then aggregate
python -m src.aggregate_human --human_csv results\human.csv

# 7) Plots
python -m src.plots --overall results\results.csv
python -m src.plots --by_length results\acc_by_length.csv
python -m src.plots --by_template results\acc_by_template.csv --top_k 12
python -m src.plots --overlay_len_model results\acc_by_length.csv --overlay_len_human results\human_acc_by_length.csv
python -m src.plots --overlay_tpl_model results\acc_by_template.csv --overlay_tpl_human results\human_acc_by_template.csv --top_k 12

