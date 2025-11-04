#!/usr/bin/env bash
set -e
python -m src.data --config configs/base.yaml
python -m src.train --config configs/lstm.yaml
python -m src.train --config configs/transformer.yaml
python -m src.train --config configs/op_bottleneck.yaml
python -m src.evaluate --config configs/base.yaml
python -m src.plots --config configs/base.yaml
pandoc docs/report.md -o docs/report.pdf || true

