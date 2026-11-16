"""
M7 — Dual-Branch Conv-Mamba (Official mamba-ssm package)
=========================================================
Run these two cells FIRST before this file:
  !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  !pip install mamba-ssm causal-conv1d
  # Then restart kernel

Architecture:
  Input (B, 900, 2)
  ├─ RRI  branch : Conv1d(16)->BN->ReLU -> Conv1d(24,s=2)->BN->ReLU->MaxPool(3)
  │                -> Conv1d(32)->BN->ReLU->MaxPool(5)           -> (B, 32, 30)
  ├─ Ampl branch : (same)                                        -> (B, 32, 30)
  ├─ Concatenate                                                 -> (B, 64, 30)
  ├─ Transpose                                                   -> (B, 30, 64)
  ├─ ECABlock(3)                  channel recalibration
  ├─ PositionalEncoding(30, 64)   temporal order
  ├─ Mamba(d_model=64) x2         official selective SSM
  ├─ GAP + GMP -> Concat          -> (B, 128)
  ├─ Linear(128, 64) -> Dropout
  └─ Linear(64, 2)

Training:
  - Focal loss (class-weighted, gamma=2)
  - Intra-class MixUp (50%) + cross-class (50%)
  - Cosine LR with 5-epoch warmup
  - Gradient clipping (norm=1.0)
  - Best model saved by val F1
  - TTA x10 at inference
  - Threshold search on val set
"""

# =========================
# INSTALL CHECK
# =========================

try:
    from mamba_ssm import Mamba
    print("mamba-ssm imported successfully")
except ImportError:
    raise ImportError(
        "\n\nmamba-ssm not found. Run these cells first:\n"
        "  !pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu126\n"
        "  !pip install mamba-ssm causal-conv1d\n"
        "Then restart the kernel."
    )

import pickle
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mamba_ssm import Mamba
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

base_dir   = "/kaggle/input/datasets/chaimahannachi/apneaecg"
out_dir    = "/kaggle/working"
os.makedirs(out_dir, exist_ok=True)

ir         = 3
before     = 2
after      = 2

BATCH_SIZE = 128
EPOCHS     = 120
SEED       = 42
TTA_RUNS   = 10
TTA_SIGMA  = 0.005
DROPOUT    = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
