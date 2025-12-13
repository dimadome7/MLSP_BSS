# exit on error
set -e

echo '>>> creating virtual environment...'
python3 -m venv .venv

echo '>>> activating virtual environment...'
source .venv/bin/activate

echo '>>> installing requirements...'
pip install --upgrade pip
pip install -r requirements.txt

echo '>>> running separation script...'
python3 main.py

echo '>>> done.'