"""Subject-level evaluation: per-recording AHI from segment predictions, then AHI >= 5 classification.

Place this file in  evaluation/  next to  keras_models/  (it imports ECABlock from keras_models/keras_base_model.py).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "keras_models"))
from keras_base_model import ECABlock

# -----------------------------
# Settings
# -----------------------------
model_path = "/kaggle/input/models/chaimahannachi/m8/keras/default/1/M8_GRU_custom (1).keras"
base_dir   = "/kaggle/input/datasets/chaimahannachi/apneaecg"
info_path  = "/kaggle/input/datasets/chaimahannachi/add-information/additional-information.txt"
pkl_name   = "apnea-ecg (1).pkl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ir = 3         # interpolation rate (samples/sec)
before = 2     # minutes before event
after = 2      # minutes after event


def scaler(arr):
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


# -----------------------------
# Load test data (same preprocessing as training)
# -----------------------------
def load_test_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))  # interpolation timeline
    with open(os.path.join(base_dir, pkl_name), "rb") as f:
        apnea_ecg = pickle.load(f)

    x_test = []
    for (rri_tm, rri_signal), (ampl_tm, ampl_signal) in apnea_ecg["o_test"]:
        rri_interp  = splev(tm, splrep(rri_tm,  scaler(rri_signal),  k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test.append([rri_interp, ampl_interp])

    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))  # (samples, timesteps, features)
    return x_test, apnea_ecg["groups_test"]                          # subject ID of every segment


# -----------------------------
# Reference AHI (x01-x35) from additional-information.txt
# -----------------------------
def load_true_ahi():
    records = []
    with open(info_path, "r") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) == 12 and cols[0].startswith("x"):        # valid rows, test subjects only
                records.append([cols[0], float(cols[3]) / float(cols[1]) * 60])
    df_true = pd.DataFrame(records, columns=["subject", "AHI_true"]).set_index("subject")
    df_true["AHI_binary_true"] = (df_true["AHI_true"] > 5).astype(int)
    return df_true


# -----------------------------
# Predicted AHI per subject
# -----------------------------
def predict_ahi(model, x_test, subjects_test):
    y_score = model.predict(x_test)[:, 1]
    df = pd.DataFrame({"subject": subjects_test,
                       "y_score": y_score,
                       "y_pred": (y_score > 0.5).astype(int)})
    ahi = np.round(df.groupby("subject")["y_pred"].mean() * 60)    # events per hour
    return pd.DataFrame({"AHI_pred": ahi, "AHI_binary_pred": (ahi >= 5).astype(int)})


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    model = load_model(model_path, custom_objects={"ECABlock": ECABlock, "Custom>ECABlock": ECABlock})
    model.summary()

    x_test, subjects_test = load_test_data()
    df_final = load_true_ahi().join(predict_ahi(model, x_test, subjects_test))
    print(df_final)

    y_true, y_pred = df_final["AHI_binary_true"], df_final["AHI_binary_pred"]
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", cm)
    print("Sensitivity:", TP / (TP + FN))
    print("Specificity:", TN / (TN + FP))
