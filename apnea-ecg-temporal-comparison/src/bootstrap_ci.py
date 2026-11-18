"""
Bootstrap confidence intervals for the segment-level comparison (Table 2).

Step 1  Predictions. For each Keras model, the trained weights are loaded, the
        test set is predicted with test-time augmentation (10 noisy passes,
        sigma = 0.005, probabilities averaged) and the apnea probabilities are
        saved to results/predictions/<model>.npz. A file that already exists is
        reused, so the predictions behind a given set of intervals stay fixed.
Step 2  Intervals. Each model is evaluated at its validation-selected threshold.
        95% percentile intervals (2.5th and 97.5th percentiles) are computed
        from 1,000 resamples with replacement, in two ways:
          segment    individual test segments are resampled (as in the paper);
          recording  whole test recordings are resampled, which keeps the
                     correlated, overlapping windows of a recording together.
        The same resamples are used for all models, so paired differences from
        CNN+GRU are also reported.

Run from inside src/:   python bootstrap_ci.py
                        python bootstrap_ci.py --skip-predict   (use saved .npz only)

Needs:
  dataset/apnea-ecg.pkl          (from preprocess_apnea_ecg.py)
  weights/<model>.weights.h5     (from the train_cnn_*.py scripts)
Writes:
  results/predictions/<model>.npz
  results/bootstrap_ci.csv
  results/bootstrap_paired_vs_gru.csv

CNN+Mamba is a PyTorch model. Save its test predictions once, in the same order
as the test set of apnea-ecg.pkl, with

    np.savez("results/predictions/cnn_mamba.npz", y_true=y_test, prob=apnea_prob)

and this script includes it. Without that file the model is skipped and reported.
"""

import argparse
import importlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from sklearn.metrics import roc_auc_score

from paths import OUT_DIR, WEIGHTS_DIR, PKL_PATH

# ── Settings ──────────────────────────────────────────────────────────────────
N_BOOT = 1000
SEED = 42
TTA_RUNS = 10
TTA_SIGMA = 0.005
IR, BEFORE, AFTER = 3, 2, 2
REFERENCE = "CNN+GRU"
METRICS = ["ACC", "SN", "SP", "F1", "AUC"]

# name: (file stem, training module or None, validation-selected threshold)
MODELS = {
    "CNN+Transformer": ("cnn_transformer", "train_cnn_transformer", 0.50),
    "CNN+Mamba":       ("cnn_mamba",       None,                    0.54),
    "CNN+LSTM":        ("cnn_lstm",        "train_cnn_lstm",        0.60),
    "CNN+GRU":         ("cnn_gru",         "train_cnn_gru",         0.50),
    "CNN+BiGRU":       ("cnn_bigru",       "train_cnn_bigru",       0.47),
    "CNN+BiLSTM":      ("cnn_bilstm",      "train_cnn_bilstm",      0.54),
}

PRED_DIR = OUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ── Test data ─────────────────────────────────────────────────────────────────
def scaler(a):
    a = np.asarray(a)
    return (a - a.min()) / (a.max() - a.min())


def load_test_set(with_inputs):
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["y_test"]).astype(int)
    records = np.asarray([str(g) for g in data["groups_test"]])
    x = None
    if with_inputs:
        tm = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))
        x = []
        for (rri_tm, rri), (ampl_tm, ampl) in data["o_test"]:
            x.append([splev(tm, splrep(rri_tm, scaler(rri), k=3), ext=1),
                      splev(tm, splrep(ampl_tm, scaler(ampl), k=3), ext=1)])
        x = np.asarray(x, dtype="float32").transpose((0, 2, 1))
    assert len(y) == len(records)
    return x, y, records


# ── Step 1: predictions ───────────────────────────────────────────────────────
def predict_tta(model, x):
    rng = np.random.default_rng(SEED)
    total = np.zeros(len(x), dtype="float64")
    for k in range(TTA_RUNS):
        noise = rng.normal(0, TTA_SIGMA, x.shape).astype("float32")
        total += model.predict(x + noise, batch_size=512, verbose=0)[:, 1]
        print(f"    TTA pass {k + 1}/{TTA_RUNS}", flush=True)
    return total / TTA_RUNS


def make_predictions(y_test):
    x_test = None
    for name, (stem, module, _) in MODELS.items():
        target = PRED_DIR / f"{stem}.npz"
        if target.is_file():
            print(f"[predict] {name}: reusing {target.name}")
            continue
        if module is None:
            print(f"[predict] {name}: no {target.name}; see the note at the top of this file")
            continue
        weights = WEIGHTS_DIR / f"{stem}.weights.h5"
        if not weights.is_file():
            print(f"[predict] {name}: weights not found ({weights}); skipped")
            continue
        if x_test is None:
            print("[predict] building the test inputs ...", flush=True)
            x_test, _, _ = load_test_set(with_inputs=True)
        print(f"[predict] {name}: loading {weights.name}", flush=True)
        model = importlib.import_module(module).create_model((900, 2))
        model.load_weights(weights)
        prob = predict_tta(model, x_test)
        np.savez(target, y_true=y_test, prob=prob)
        print(f"[predict] {name}: saved {target.name}")


# ── Step 2: metrics and intervals ─────────────────────────────────────────────
def metrics(y, prob, threshold):
    pred = prob >= threshold
    pos = y == 1
    tp = np.sum(pred & pos); fn = np.sum(~pred & pos)
    tn = np.sum(~pred & ~pos); fp = np.sum(pred & ~pos)
    sn = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    auc = roc_auc_score(y, prob) if 0 < pos.sum() < len(y) else np.nan
    return 100 * np.array([(tp + tn) / len(y), sn, sp, f1, auc])


def resample_indices(scheme, records, rng):
    n = len(records)
    if scheme == "segment":
        return rng.integers(0, n, n)
    names = np.unique(records)
    members = {r: np.flatnonzero(records == r) for r in names}
    drawn = rng.choice(names, size=len(names), replace=True)
    return np.concatenate([members[r] for r in drawn])


def bootstrap(scheme, y, records, probs):
    """Returns {model: array (N_BOOT, 5)}; the same resamples serve all models."""
    rng = np.random.default_rng(SEED)
    out = {name: np.empty((N_BOOT, len(METRICS))) for name in probs}
    start = time.time()
    for b in range(N_BOOT):
        idx = resample_indices(scheme, records, rng)
        for name, prob in probs.items():
            out[name][b] = metrics(y[idx], prob[idx], MODELS[name][2])
        if (b + 1) % 100 == 0:
            print(f"  [{scheme}] resample {b + 1}/{N_BOOT}  ({time.time() - start:.0f} s)",
                  flush=True)
    return out


def interval(values):
    return np.nanpercentile(values, [2.5, 97.5], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-predict", action="store_true",
                        help="use the saved .npz predictions only")
    args = parser.parse_args()

    _, y_test, records = load_test_set(with_inputs=False)
    print(f"Test segments: {len(y_test)}   recordings: {len(np.unique(records))}")
    if not args.skip_predict:
        make_predictions(y_test)

    probs = {}
    for name, (stem, _, _) in MODELS.items():
        path = PRED_DIR / f"{stem}.npz"
        if not path.is_file():
            print(f"[intervals] {name}: no predictions; not included")
            continue
        saved = np.load(path)
        if len(saved["prob"]) != len(y_test) or not np.array_equal(
                saved["y_true"].astype(int), y_test):
            sys.exit(f"{path.name}: labels do not match the test set of {PKL_PATH.name}")
        probs[name] = saved["prob"].astype("float64")
    if not probs:
        sys.exit("No predictions available.")

    point = {name: metrics(y_test, prob, MODELS[name][2]) for name, prob in probs.items()}
    rows, paired_rows = [], []
    for scheme in ("segment", "recording"):
        print(f"\nBootstrap by {scheme} ({N_BOOT} resamples, {len(probs)} models)", flush=True)
        boot = bootstrap(scheme, y_test, records, probs)
        for name in probs:
            low, high = interval(boot[name])
            for m, metric in enumerate(METRICS):
                rows.append([scheme, name, metric, MODELS[name][2],
                             point[name][m], low[m], high[m]])
        if REFERENCE in probs:
            for name in probs:
                if name == REFERENCE:
                    continue
                diff = boot[name] - boot[REFERENCE]
                low, high = interval(diff)
                for m, metric in enumerate(METRICS):
                    paired_rows.append([scheme, name, metric,
                                        point[name][m] - point[REFERENCE][m], low[m], high[m]])

    ci = pd.DataFrame(rows, columns=["resampling_unit", "model", "metric", "threshold",
                                     "value", "ci_low", "ci_high"]).round(2)
    ci.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'bootstrap_ci.csv'}")
    if paired_rows:
        paired = pd.DataFrame(paired_rows, columns=["resampling_unit", "model", "metric",
                                                    f"difference_from_{REFERENCE}",
                                                    "ci_low", "ci_high"]).round(2)
        paired.to_csv(OUT_DIR / "bootstrap_paired_vs_gru.csv", index=False)
        print(f"Saved {OUT_DIR / 'bootstrap_paired_vs_gru.csv'}")

    for scheme in ("segment", "recording"):
        print(f"\n95% intervals, resampling by {scheme}")
        print(f"  {'Model':<17s}" + "".join(f"{m:>22s}" for m in METRICS))
        for name in probs:
            part = ci[(ci.resampling_unit == scheme) & (ci.model == name)]
            cells = [f"{r.value:.2f} [{r.ci_low:.2f}-{r.ci_high:.2f}]" for r in part.itertuples()]
            print(f"  {name:<17s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
