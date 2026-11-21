"""
Record-level evaluation of the CNN+GRU model (single prediction pass, no TTA).
Prediction -> apnea burden per record -> comparison with the reference AHI.
Rule (as in the paper): a record is apnea if AHI >= 5 events per hour (no rounding).

Run from inside src/:   python subject_level_evaluation.py

Needs:
  weights/cnn_gru.weights.h5            (from train_cnn_gru.py)
  dataset/apnea-ecg.pkl                 (from preprocess_apnea_ecg.py)
  dataset/additional-information.txt    (from the PhysioNet Apnea-ECG download)
Writes:
  results/record_level_gru.csv
"""

import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, accuracy_score

from paths import DATA_DIR, OUT_DIR, WEIGHTS_DIR, PKL_PATH
from train_cnn_gru import create_model   # same architecture as in training

# ── Settings ──────────────────────────────────────────────────────────────────
weights_path = WEIGHTS_DIR / "cnn_gru.weights.h5"
info_path = DATA_DIR / "additional-information.txt"

ir, before, after = 3, 2, 2
# Decision threshold of the GRU model.
THRESHOLD = 0.49
CUTOFF = 5


# ── Model ─────────────────────────────────────────────────────────────────────
model = create_model((900, 2))
model.load_weights(weights_path)


# ── Test data ─────────────────────────────────────────────────────────────────
def scaler(a):
    a = np.array(a)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
with open(PKL_PATH, "rb") as f:
    apnea_ecg = pickle.load(f)

x_test = []
for (rri_tm, rri), (ampl_tm, ampl) in apnea_ecg["o_test"]:
    x_test.append([splev(tm, splrep(rri_tm, scaler(rri), k=3), ext=1),
                   splev(tm, splrep(ampl_tm, scaler(ampl), k=3), ext=1)])
x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))

y_test = np.array(apnea_ecg["y_test"])
subjects = np.array(apnea_ecg["groups_test"])
assert len(x_test) == len(y_test) == len(subjects)

# ── Prediction (single pass, no TTA) ─────────────────────────────────────────
y_score = model.predict(x_test, batch_size=512, verbose=0)[:, 1]
y_pred = (y_score >= THRESHOLD).astype(int)

# ── Predicted burden per record (apnea minutes per hour) ─────────────────────
seg = pd.DataFrame({"subject": subjects, "y_pred": y_pred})
AHI_pred = seg.groupby("subject")["y_pred"].mean() * 60
AHI_pred.index = [s.lower() if str(s).lower().startswith("x") else f"x{int(float(s)):02d}"
                  for s in AHI_pred.index]

# ── Reference AHI (events per hour) ──────────────────────────────────────────
rows = []
with open(info_path) as f:
    for line in f:
        cols = line.split("\t")
        if len(cols) != 12:
            cols = line.split()
        if len(cols) == 12 and cols[0].lower().startswith("x"):
            rows.append([cols[0].lower(), float(cols[3]) / float(cols[1]) * 60])

df = pd.DataFrame(rows, columns=["subject", "AHI_true"]).set_index("subject")
df["AHI_pred"] = AHI_pred
df["true_class"] = (df["AHI_true"] >= CUTOFF).astype(int)
df["pred_class"] = (df["AHI_pred"] >= CUTOFF).astype(int)
print(df)
print("Records compared:", len(df))          # should be 35

# ── Metrics ──────────────────────────────────────────────────────────────────
TN, FP, FN, TP = confusion_matrix(df["true_class"], df["pred_class"], labels=[0, 1]).ravel()
print("\nthreshold:", THRESHOLD, " cutoff: AHI >=", CUTOFF)
print("Confusion matrix [TN FP / FN TP]:", TN, FP, FN, TP)
print("Accuracy   :", accuracy_score(df["true_class"], df["pred_class"]))
print("Sensitivity:", TP / (TP + FN))
print("Specificity:", TN / (TN + FP))
print("MAE        :", (df["AHI_pred"] - df["AHI_true"]).abs().mean())

df.to_csv(OUT_DIR / "record_level_gru.csv")
