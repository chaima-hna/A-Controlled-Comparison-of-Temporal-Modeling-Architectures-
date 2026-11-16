
 

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

# ── Config ────────────────────────────────────────────────────────────────────

base_dir = "/kaggle/input/datasets/chaimahannachi/apneaecg"
out_dir  = "/kaggle/working"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ir     = 3
before = 2
after  = 2

BATCH_SIZE   = 128
EPOCHS       = 120
SEED         = 42
TTA_RUNS     = 10
TTA_SIGMA    = 0.005
