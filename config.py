"""Portable paths; outputs live in RUN_DIR, separate from source code."""
import hashlib
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RUN_DIR', CODE / 'runs/qwen3vl8b')).expanduser().resolve()
MODEL = Path(os.environ.get('EVAL_MODEL', CODE / 'assets/Qwen3-VL-8B-Instruct')).expanduser().resolve()
VSTAR_DIR = Path(os.environ.get('VSTAR_DIR', CODE / 'assets/vstar_bench')).expanduser().resolve()
RAW_DATA = Path(os.environ.get('RAW_DATA', CODE / 'assets/DeepEyes-Datasets-47k')).expanduser().resolve()
os.environ.setdefault('EVAL_MODEL', str(MODEL))
ROOT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()
