#!/usr/bin/env bash
set -euo pipefail

echo ">>> checking python3.11..."
command -v python3.11 >/dev/null 2>&1 || { echo "ERROR: python3.11 not found in PATH"; exit 1; }
python3.11 --version

echo ">>> recreating virtual environment..."
rm -rf .venv
python3.11 -m venv .venv

echo ">>> activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> python in venv:"
which python
python --version

echo ">>> installing requirements..."
export PIP_ONLY_BINARY=:all:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ">>> running separation script..."
python main.py

echo ">>> done."
