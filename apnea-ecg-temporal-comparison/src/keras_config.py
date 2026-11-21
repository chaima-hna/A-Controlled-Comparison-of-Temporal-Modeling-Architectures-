import pickle
import keras
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import (
    Conv1D, Dense, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Input, LayerNormalization, Dropout, BatchNormalization,
    Activation, Concatenate, MaxPooling1D,
)
from keras.models import Model, load_model
from keras.utils import to_categorical

from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.saving import register_keras_serializable

from paths import DATA_DIR, OUT_DIR, WEIGHTS_DIR, PKL_PATH

# ── Config ────────────────────────────────────────────────────────────────────

# Portable paths (see paths.py). Old names kept so the other scripts still work.
base_dir = str(DATA_DIR)
out_dir  = str(OUT_DIR)          # logs, figures
weights_dir = str(WEIGHTS_DIR)  # trained models
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ir     = 3
before = 2
after  = 2

BATCH_SIZE   = 128
EPOCHS       = 120
SEED         = 42
