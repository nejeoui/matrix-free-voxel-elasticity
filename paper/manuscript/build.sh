#!/bin/sh
# Regenerate every figure, table and number from verified evidence, then compile.
set -e
cd "$(dirname "$0")"
PYTHONDONTWRITEBYTECODE=1 ${PY:-python3} ../../analysis/paper_numbers.py > /dev/null
PYTHONDONTWRITEBYTECODE=1 ${PY:-python3} ../../analysis/roofline.py > /dev/null
PYTHONDONTWRITEBYTECODE=1 ${PY:-python3} make_figures.py
