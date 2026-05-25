#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" experiments.py
tectonic report.tex
cp report.pdf ms2_report.pdf

echo "Regenerated figures/summary.json, figures/*.png, report.pdf, and ms2_report.pdf"
