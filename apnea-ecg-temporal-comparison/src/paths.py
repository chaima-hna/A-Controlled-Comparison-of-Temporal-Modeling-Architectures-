"""Shared, portable paths. No heavy imports, so preprocessing can use it without TensorFlow.

Override with environment variables, e.g. on Kaggle:
    APNEA_DATA_DIR=/kaggle/input/datasets/chaimahannachi/apneaecg
    APNEA_OUT_DIR=/kaggle/working/results
    APNEA_WEIGHTS_DIR=/kaggle/working/weights
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in src/

# Raw PhysioNet Apnea-ECG files (a01.dat, a01.hea, a01.apn, ..., event-2-answers)
# and the preprocessed pickle both live here.
DATA_DIR = Path(os.environ.get("APNEA_DATA_DIR", REPO_ROOT / "dataset"))

# Logs, CSVs, figures
OUT_DIR = Path(os.environ.get("APNEA_OUT_DIR", REPO_ROOT / "results"))

# Trained models
WEIGHTS_DIR = Path(os.environ.get("APNEA_WEIGHTS_DIR", REPO_ROOT / "weights"))

PKL_NAME = "apnea-ecg.pkl"
PKL_PATH = DATA_DIR / PKL_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
