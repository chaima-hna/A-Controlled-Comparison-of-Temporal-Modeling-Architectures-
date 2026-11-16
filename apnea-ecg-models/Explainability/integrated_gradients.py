import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Must be set before importing Keras

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from scipy.stats import ttest_rel, wilcoxon
from scipy.interpolate import splev, splrep
from sklearn.metrics import roc_auc_score

import keras
from keras.models import load_model
from keras.saving import register_keras_serializable
from keras.layers import Conv1D


# ============================================================
# CONFIGURATION
# ============================================================
# Expected project structure:
#
# Sleep-Apnea-Detection/
# ├── models/
# │   └── M8_GRU_custom.keras
# ├── data/
# │   └── apnea-ecg.pkl
# ├── results/
# └── src/
#     └── explainability/
#         └── integrated_gradients.py
#
# Run from the project root with:
# python -m src.explainability.integrated_gradients

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAVE_DIR = PROJECT_ROOT / "results" / "ig_figures"
MODEL_PATH = PROJECT_ROOT / "models" / "M8_GRU_custom.keras"
DATA_PATH = PROJECT_ROOT / "data" / "apnea-ecg.pkl"

N_STEPS = 50
N_PER_CLASS = 25
SEED = 42
TOP_FRAC = 0.10
SAMPLE_IDX = None
EPS = 1e-7

BATCH_SIZE = 256
ABLATION_BATCH_SIZE = 512

SAVE_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def check_input_files():
    """Check that the model and dataset exist before starting."""
    missing = []
    if not MODEL_PATH.is_file():
        missing.append(f"Trained model not found: {MODEL_PATH}")
    if not DATA_PATH.is_file():
        missing.append(f"Dataset pickle not found: {DATA_PATH}")

    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(
            f"{message}\n\n"
            "Place M8_GRU_custom.keras in the project's models/ folder "
            "and apnea-ecg.pkl in the data/ folder."
        )


# ============================================================
# 1. CUSTOM LAYER AND MODEL
# ============================================================
@register_keras_serializable(package="Custom")
class ECABlock(keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(
            1,
            kernel_size,
            padding="same",
            use_bias=False
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)
        gap = tf.expand_dims(tf.reduce_mean(x, axis=1), axis=-1)
        weights = tf.nn.sigmoid(self.conv(gap))
        weights = tf.expand_dims(tf.squeeze(weights, axis=-1), axis=1)
        return tf.identity(x * weights)

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def load_trained_model():
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "ECABlock": ECABlock,
            "Custom>ECABlock": ECABlock,
        },
    )

    if len(model.output_shape) != 2:
        raise ValueError(
            f"Expected a model with a 2D output shape; got {model.output_shape}."
        )

    output_units = model.output_shape[-1]
    if output_units not in (1, 2):
        raise ValueError(
            f"Expected one or two output units; got {output_units}."
        )

    return model, output_units


# ============================================================
# 2. LOAD AND PREPARE THE TEST DATA
# ============================================================
IR = 3  # Interpolation frequency: 3 Hz


def minmax_scaler(values):
    values = np.asarray(values)
    return (values - values.min()) / (
        values.max() - values.min() + 1e-8
    )


def encode_label(label):
    if isinstance(label, (list, tuple, np.ndarray)):
        label = np.asarray(label).ravel()[0]

    if isinstance(label, bytes):
        label = label.decode()

    value = str(label).strip().upper()

    if value in ("A", "1", "1.0", "TRUE", "APNEA"):
        return 1
    if value in ("N", "0", "0.0", "FALSE", "NORMAL"):
        return 0

    raise ValueError(f"Unknown label: {label!r}")


def load_test_data():
    with open(DATA_PATH, "rb") as file:
        apnea_ecg = pickle.load(file)

    if "o_test" not in apnea_ecg or "y_test" not in apnea_ecg:
        raise KeyError(
            "The pickle file must contain 'o_test' and 'y_test' entries."
        )

    time_axis = np.arange(0, 5 * 60, step=1 / float(IR))
    segments = []

    for (rri_time, rri_signal), (amp_time, amp_signal) in apnea_ecg["o_test"]:
        rri = splev(
            time_axis,
            splrep(rri_time, minmax_scaler(rri_signal), k=3),
            ext=1,
        )
        amplitude = splev(
            time_axis,
            splrep(amp_time, minmax_scaler(amp_signal), k=3),
            ext=1,
        )
        segments.append([rri, amplitude])

    x_test = np.asarray(segments, dtype="float32").transpose((0, 2, 1))
    y_test = np.asarray(
        [encode_label(label) for label in apnea_ecg["y_test"]],
        dtype=int,
    )

    if len(x_test) != len(y_test):
        raise ValueError(
            f"Number of test segments ({len(x_test)}) does not match "
            f"number of labels ({len(y_test)})."
        )

    if x_test.ndim != 3 or x_test.shape[1:] != (900, 2):
        raise ValueError(
            f"Expected test input shape (N, 900, 2); got {x_test.shape}."
        )

    if not np.isfinite(x_test).all():
        raise ValueError("The test data contains NaN or infinite values.")

    return apnea_ecg, x_test, y_test


# ============================================================
# 3. PREDICTIONS AND APNEA OUTPUT
# ============================================================
def get_apnea_output_index(predictions, y_test, output_units):
    """
    Determine the output column associated with apnea.

    IMPORTANT:
    For strict evaluation without test-label-informed choices, set the
    apnea output index from the model's training configuration instead
    of inferring it from test-set labels.
    """
    if output_units == 1:
        return 1

    if predictions.shape[1] != 2:
        raise ValueError("A two-output model must return two columns.")

    # This reproduces the original article script's output-mapping rule.
    mean_for_class_1 = predictions[y_test == 1, 1].mean()
    mean_for_class_0 = predictions[y_test == 0, 1].mean()
    return 1 if mean_for_class_1 >= mean_for_class_0 else 0


def get_apnea_probability(predictions, output_units, apnea_output_index):
    if output_units == 2:
        return predictions[:, apnea_output_index]

    if apnea_output_index == 1:
        return predictions[:, 0]

    return 1.0 - predictions[:, 0]


# ============================================================
# 4. INTEGRATED GRADIENTS
# ============================================================
ALPHAS = None


def make_target_logodds(output_units, apnea_output_index):
    def target_logodds(outputs):
        if output_units == 1:
            if apnea_output_index == 1:
                apnea_probability = outputs[:, 0]
            else:
                apnea_probability = 1.0 - outputs[:, 0]
            normal_probability = 1.0 - apnea_probability
        else:
            apnea_probability = outputs[:, apnea_output_index]
            normal_probability = outputs[:, 1 - apnea_output_index]

        return (
            tf.math.log(apnea_probability + EPS)
            - tf.math.log(normal_probability + EPS)
        )

    return target_logodds


def calculate_integrated_gradients(model, target_logodds, x, alphas):
    baseline = np.zeros_like(x)

    interpolated = (
        tf.constant(baseline, dtype=tf.float32)[None, ...]
        + alphas[:, None, None]
        * (
            tf.constant(x, dtype=tf.float32)
            - tf.constant(baseline, dtype=tf.float32)
        )[None, ...]
    )

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        outputs = target_logodds(
            model(interpolated, training=False)
        )

    gradients = tape.gradient(outputs, interpolated)

    if gradients is None:
        raise RuntimeError(
            "Could not calculate gradients. Check the model output "
            "and target-log-odds function."
        )

    # Trapezoidal approximation across integration steps.
    average_gradients = (gradients[:-1] + gradients[1:]) / 2.0
    average_gradients = tf.reduce_mean(average_gradients, axis=0)

    return (
        average_gradients.numpy() * (x - baseline)
    ).astype("float32")


def calculate_all_attributions(model, target_logodds, x_test):
    cache_path = SAVE_DIR / f"ig_article_N{len(x_test)}_S{N_STEPS}.npy"

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == x_test.shape:
            print(f"Loaded cached attributions: {cache_path}")
            return cached
        print("Cached attribution shape does not match; recalculating.")

    alphas = tf.linspace(0.0, 1.0, N_STEPS + 1)
    all_attributions = np.zeros_like(x_test, dtype="float32")

    for index in range(len(x_test)):
        all_attributions[index] = calculate_integrated_gradients(
            model,
            target_logodds,
            x_test[index],
            alphas,
        )

        if (index + 1) % 100 == 0 or index + 1 == len(x_test):
            print(
                f"Integrated Gradients: {index + 1}/{len(x_test)}",
                end="\r",
                flush=True,
            )

    np.save(cache_path, all_attributions)
    print(f"\nSaved attributions: {cache_path}")
    return all_attributions


# ============================================================
# 5. QUANTITATIVE EXPLAINABILITY MEASURES
# ============================================================
def gini_rows(matrix):
    sorted_values = np.sort(np.abs(matrix), axis=1)
    n_features = sorted_values.shape[1]
    cumulative = np.cumsum(sorted_values, axis=1)

    return (
        n_features
        + 1
        - 2 * cumulative.sum(axis=1)
        / (cumulative[:, -1] + 1e-12)
    ) / n_features


def row_correlation(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)

    denominator = np.sqrt(
        (a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)
    )

    return (a * b).sum(axis=1) / np.where(
        denominator > 1e-12,
        denominator,
        np.nan,
    )


def calculate_segment_metrics(probabilities, y_test):
    predictions = (probabilities > 0.5).astype(int)

    tp = int(((predictions == 1) & (y_test == 1)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / len(y_test)

    auc = (
        roc_auc_score(y_test, probabilities)
        if len(np.unique(y_test)) == 2
        else float("nan")
    )

    return (
        accuracy * 100,
        sensitivity * 100,
        specificity * 100,
        f1 * 100,
        auc * 100,
    )


def report_explainability_measures(
    all_attributions,
    x_test,
    y_test,
):
    absolute_attributions = np.abs(all_attributions)
    apnea_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]

    if len(apnea_indices) == 0 or len(normal_indices) == 0:
        raise ValueError("The test set must contain both classes.")

    print("\n" + "=" * 65)
    print(
        f"INTEGRATED GRADIENTS MEASURES "
        f"(all {len(y_test)} test segments, {N_STEPS} steps)"
    )
    print("=" * 65)

    print("\nMean absolute attribution")
    print(
        f"  Apnea : RRI {absolute_attributions[apnea_indices, :, 0].mean():.4f}"
        f" | Amplitude {absolute_attributions[apnea_indices, :, 1].mean():.4f}"
    )
    print(
        f"  Normal: RRI {absolute_attributions[normal_indices, :, 0].mean():.4f}"
        f" | Amplitude {absolute_attributions[normal_indices, :, 1].mean():.4f}"
    )

    print("\nGini coefficient (mean over segments)")
    print(
        f"  G_RRI = {gini_rows(all_attributions[:, :, 0]).mean():.4f}"
        f" | G_Ampl = {gini_rows(all_attributions[:, :, 1]).mean():.4f}"
    )

    rri_correlation = row_correlation(
        absolute_attributions[:, :, 0],
        np.abs(np.gradient(x_test[:, :, 0], axis=1)) * IR,
    )
    amp_correlation = row_correlation(
        absolute_attributions[:, :, 1],
        np.abs(np.gradient(x_test[:, :, 1], axis=1)) * IR,
    )

    print("\nPearson correlation of |attribution| with |dx/dt|")
    print(
        f"  R_RRI = {np.nanmean(rri_correlation):.4f}"
        f" | R_Ampl = {np.nanmean(amp_correlation):.4f}"
    )

    segment_rri = absolute_attributions[:, :, 0].mean(axis=1)
    segment_amp = absolute_attributions[:, :, 1].mean(axis=1)

    contribution_sum = segment_rri.mean() + segment_amp.mean()
    if contribution_sum <= 0:
        raise ValueError("Total attribution is zero; channel shares undefined.")

    share_rri = segment_rri.mean() / contribution_sum * 100

    if np.allclose(segment_rri, segment_amp):
        t_statistic, p_value = 0.0, 1.0
    else:
        t_statistic, p_value = ttest_rel(
            segment_rri,
            segment_amp,
            nan_policy="omit",
        )

    print("\nGlobal channel contribution")
    print(
        f"  RRI {share_rri:.2f}%"
        f" | R-amplitude {100 - share_rri:.2f}%"
    )
    print(
        f"  Paired t-test: t = {t_statistic:.4f}, "
        f"p = {p_value:.3e}"
    )

    return absolute_attributions, share_rri


# ============================================================
# 5B. CHANNEL ABLATION AND DELETION TEST
# ============================================================
def run_channel_ablation(
    model,
    x_test,
    y_test,
    output_units,
    apnea_output_index,
    full_probabilities,
):
    print(
        "\nChannel ablation "
        "(one channel replaced by its segment mean; threshold 0.5)"
    )
    print(
        f"  {'Input':<24s}{'ACC':>8s}{'SN':>8s}"
        f"{'SP':>8s}{'F1':>8s}{'AUC':>8s}"
    )

    results = {"Full model": full_probabilities}

    for channel, name in [
        (0, "RRI removed"),
        (1, "R-amplitude removed"),
    ]:
        modified_x = x_test.copy()
        modified_x[:, :, channel] = x_test[:, :, channel].mean(
            axis=1,
            keepdims=True,
        )

        predictions = model.predict(
            modified_x,
            batch_size=ABLATION_BATCH_SIZE,
            verbose=0,
        )
        results[name] = get_apnea_probability(
            predictions,
            output_units,
            apnea_output_index,
        )

    for name, probabilities in results.items():
        accuracy, sensitivity, specificity, f1, auc = (
            calculate_segment_metrics(probabilities, y_test)
        )
        print(
            f"  {name:<24s}{accuracy:8.2f}{sensitivity:8.2f}"
            f"{specificity:8.2f}{f1:8.2f}{auc:8.2f}"
        )


def run_deletion_test(
    model,
    target_logodds,
    x_test,
    y_test,
    absolute_attributions,
    full_probabilities,
    apnea_ecg,
):
    print("\nDeletion test on correctly classified segments")
    print(
        "Top-attributed points are replaced by the segment mean "
        "and compared with random points."
    )

    rng = np.random.default_rng(SEED)
    correctly_classified = np.where(
        (full_probabilities > 0.5).astype(int) == y_test
    )[0]

    if len(correctly_classified) == 0:
        print("No correctly classified segments; deletion test skipped.")
        return

    def predict_logodds(x):
        predictions = model.predict(
            x,
            batch_size=ABLATION_BATCH_SIZE,
            verbose=0,
        )
        return target_logodds(tf.constant(predictions)).numpy()

    selected_x = x_test[correctly_classified]
    baseline_logodds = predict_logodds(selected_x)

    attribution_flat = absolute_attributions[
        correctly_classified
    ].reshape(len(correctly_classified), -1)

    signal_flat = selected_x.reshape(len(correctly_classified), -1)

    # Two channels, each with 900 time points.
    segment_means = selected_x.mean(axis=1)
    mean_flat = np.tile(
        segment_means[:, None, :],
        (1, 900, 1),
    ).reshape(len(correctly_classified), -1)

    direction = np.where(
        y_test[correctly_classified] == 1,
        1.0,
        -1.0,
    )

    apnea_mask = y_test[correctly_classified] == 1
    normal_mask = y_test[correctly_classified] == 0

    record_ids = None
    if "groups_test" in apnea_ecg:
        all_record_ids = np.asarray(apnea_ecg["groups_test"])
        if len(all_record_ids) == len(y_test):
            record_ids = all_record_ids[correctly_classified]

    def calculate_shift(scores, number_to_remove):
        top_indices = np.argpartition(
            scores,
            -number_to_remove,
            axis=1,
        )[:, -number_to_remove:]

        mask = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(mask, top_indices, True, axis=1)

        modified_flat = np.where(mask, mean_flat, signal_flat)
        modified_x = modified_flat.reshape(-1, 900, 2)

        modified_logodds = predict_logodds(modified_x)

        # Positive values indicate movement toward the wrong class.
        return direction * (baseline_logodds - modified_logodds)

    print(
        f"  n = {len(correctly_classified)} "
        f"(apnea {apnea_mask.sum()}, normal {normal_mask.sum()})"
    )

    for fraction in [0.05, 0.10, 0.20, 0.30]:
        number_to_remove = max(
            1,
            int(fraction * attribution_flat.shape[1]),
        )

        top_shift = calculate_shift(
            attribution_flat,
            number_to_remove,
        )

        random_shifts = np.mean(
            [
                calculate_shift(
                    rng.random(attribution_flat.shape),
                    number_to_remove,
                )
                for _ in range(3)
            ],
            axis=0,
        )

        apnea_top = (
            top_shift[apnea_mask].mean()
            if apnea_mask.any()
            else float("nan")
        )
        apnea_random = (
            random_shifts[apnea_mask].mean()
            if apnea_mask.any()
            else float("nan")
        )
        normal_top = (
            top_shift[normal_mask].mean()
            if normal_mask.any()
            else float("nan")
        )
        normal_random = (
            random_shifts[normal_mask].mean()
            if normal_mask.any()
            else float("nan")
        )

        message = (
            f"  Remove {int(fraction * 100):2d}%: "
            f"APNEA top {apnea_top:.3f} vs random {apnea_random:.3f} | "
            f"NORMAL top {normal_top:.3f} vs random {normal_random:.3f} | "
            f"top > random in {100 * np.mean(top_shift > random_shifts):.1f}%"
            " of segments"
        )

        if record_ids is not None:
            unique_records = np.unique(record_ids)
            record_top = np.array([
                top_shift[record_ids == record].mean()
                for record in unique_records
            ])
            record_random = np.array([
                random_shifts[record_ids == record].mean()
                for record in unique_records
            ])

            if len(unique_records) >= 2 and not np.allclose(
                record_top,
                record_random,
            ):
                try:
                    wilcoxon_result = wilcoxon(
                        record_top,
                        record_random,
                    )
                    p_value = wilcoxon_result.pvalue
                except ValueError:
                    p_value = float("nan")
            else:
                p_value = float("nan")

            better_records = int((record_top > record_random).sum())
            message += (
                f" | records top > random "
                f"{better_records}/{len(unique_records)} "
                f"(Wilcoxon p={p_value:.3e})"
            )

        print(message)


# ============================================================
# 6. FIGURES
# ============================================================
def save_figure(filename):
    output_path = SAVE_DIR / filename
    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_figures(
    x_test,
    y_test,
    absolute_attributions,
    apnea_probabilities,
    share_rri,
):
    time_axis = np.arange(900) / float(IR)
    rng = np.random.default_rng(SEED)

    apnea_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]

    if len(apnea_indices) < N_PER_CLASS:
        raise ValueError(
            f"Need at least {N_PER_CLASS} apnea segments; "
            f"found {len(apnea_indices)}."
        )
    if len(normal_indices) < N_PER_CLASS:
        raise ValueError(
            f"Need at least {N_PER_CLASS} normal segments; "
            f"found {len(normal_indices)}."
        )

    selected_apnea = np.sort(
        rng.choice(apnea_indices, N_PER_CLASS, replace=False)
    )
    selected_normal = np.sort(
        rng.choice(normal_indices, N_PER_CLASS, replace=False)
    )
    selected_all = np.concatenate(
        [selected_apnea, selected_normal]
    )

    # Figure 7: Mean absolute attribution over time.
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 5),
        sharex=True,
    )

    for axis, channel, name, color in zip(
        axes,
        [0, 1],
        ["RRI", "R-amplitude"],
        ["#1f77b4", "#ff7f0e"],
    ):
        overall_mean = absolute_attributions[
            selected_all, :, channel
        ].mean(axis=0)

        apnea_mean = absolute_attributions[
            selected_apnea, :, channel
        ].mean(axis=0)

        normal_mean = absolute_attributions[
            selected_normal, :, channel
        ].mean(axis=0)

        axis.fill_between(
            time_axis,
            overall_mean,
            color=color,
            alpha=0.40,
            linewidth=0,
        )
        axis.plot(
            time_axis,
            overall_mean,
            color=color,
            linewidth=1.3,
            label=f"Overall mean (n={len(selected_all)})",
        )
        axis.plot(
            time_axis,
            apnea_mean,
            color="#b22222",
            linewidth=1.0,
            linestyle="--",
            label=f"Apnea (n={len(selected_apnea)})",
        )
        axis.plot(
            time_axis,
            normal_mean,
            color="#2e8b2e",
            linewidth=1.1,
            linestyle=":",
            label=f"Normal (n={len(selected_normal)})",
        )
        axis.set_ylabel(f"{name}\nMean |attribution|")
        axis.set_xlim(0, time_axis[-1])
        axis.set_ylim(bottom=0)
        axis.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )
        axis.spines[["top", "right"]].set_visible(False)

    axes[1].set_xlabel("Time (seconds within 5-min window)")
    plt.tight_layout()
    save_figure("fig7_ig_mean_attribution.png")

    # Figure 8: A true-positive apnea segment.
    true_positive_indices = np.where(
        (y_test == 1) & (apnea_probabilities > 0.5)
    )[0]

    if len(true_positive_indices) == 0:
        print(
            "Figure 8 skipped: no true-positive apnea segment "
            "was found at threshold 0.5."
        )
    else:
        if SAMPLE_IDX is not None:
            if SAMPLE_IDX not in true_positive_indices:
                raise ValueError(
                    f"Segment {SAMPLE_IDX} is not a true-positive apnea "
                    "segment at threshold 0.5."
                )
            selected_index = SAMPLE_IDX
        else:
            median_probability = np.median(
                apnea_probabilities[true_positive_indices]
            )
            selected_index = true_positive_indices[
                np.argmin(
                    np.abs(
                        apnea_probabilities[true_positive_indices]
                        - median_probability
                    )
                )
            ]

        print(
            f"Figure 8 segment: index {selected_index} | "
            f"ground truth APNEA | predicted APNEA | "
            f"P(apnea)={apnea_probabilities[selected_index]:.3f}"
        )

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11, 5.5),
            sharex=True,
        )

        for axis, channel, name, color in zip(
            axes,
            [0, 1],
            ["RRI", "R-amplitude"],
            ["#2b6a8f", "#d9962b"],
        ):
            attribution = absolute_attributions[
                selected_index, :, channel
            ]

            threshold = np.percentile(
                attribution,
                100 * (1 - TOP_FRAC),
            )
            important_points = attribution >= threshold

            axis.plot(
                time_axis,
                x_test[selected_index, :, channel],
                color=color,
                linewidth=1.1,
                label="Signal (min-max normalised)",
            )
            axis.vlines(
                time_axis[important_points],
                0,
                1,
                color="#c0272d",
                alpha=0.35,
                linewidth=0.8,
                label=f"Top {int(TOP_FRAC * 100)}% |attribution|",
            )
            axis.scatter(
                time_axis[important_points],
                x_test[selected_index, important_points, channel],
                s=8,
                color="#c0272d",
                zorder=3,
            )
            axis.set_ylabel(f"{name}\nNormalised value")
            axis.legend(
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="none",
            )
            axis.spines[["top", "right"]].set_visible(False)

        axes[1].set_xlabel("Time (seconds within 5-min window)")
        plt.tight_layout()
        save_figure("fig8_ig_sample_overlay.png")

    # Figure 9: Global channel contribution.
    figure, axis = plt.subplots(figsize=(4.5, 4.5))

    values = [share_rri, 100 - share_rri]
    bars = axis.bar(
        ["RRI", "R-Amplitude"],
        values,
        color=["#1f77b4", "#ff7f0e"],
        width=0.4,
    )

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontweight="semibold",
        )

    axis.set_ylabel("Global contribution (%)")
    axis.set_ylim(0, 110)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_figure("fig9_channel_share.png")


# ============================================================
# MAIN
# ============================================================
def main():
    check_input_files()

    print("Loading trained model...")
    model, output_units = load_trained_model()
    print(f"Model output shape: {model.output_shape}")

    print("\nLoading test data...")
    apnea_ecg, x_test, y_test = load_test_data()

    apnea_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]

    print(
        f"Test segments: {len(x_test)} "
        f"(apnea {len(apnea_indices)}, normal {len(normal_indices)})"
    )

    if len(apnea_indices) == 0 or len(normal_indices) == 0:
        raise ValueError("The test data must contain both classes.")

    print("\nGenerating model predictions...")
    predictions = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    apnea_output_index = get_apnea_output_index(
        predictions,
        y_test,
        output_units,
    )

    apnea_probabilities = get_apnea_probability(
        predictions,
        output_units,
        apnea_output_index,
    )

    print(f"Apnea output index: {apnea_output_index}")
    print(
        "Accuracy at threshold 0.5: "
        f"{np.mean((apnea_probabilities > 0.5) == y_test):.4f}"
    )

    target_logodds = make_target_logodds(
        output_units,
        apnea_output_index,
    )

    print("\nCalculating Integrated Gradients...")
    all_attributions = calculate_all_attributions(
        model,
        target_logodds,
        x_test,
    )

    absolute_attributions, share_rri = (
        report_explainability_measures(
            all_attributions,
            x_test,
            y_test,
        )
    )

    run_channel_ablation(
        model,
        x_test,
        y_test,
        output_units,
        apnea_output_index,
        apnea_probabilities,
    )

    run_deletion_test(
        model,
        target_logodds,
        x_test,
        y_test,
        absolute_attributions,
        apnea_probabilities,
        apnea_ecg,
    )

    print("\nGenerating figures...")
    generate_figures(
        x_test,
        y_test,
        absolute_attributions,
        apnea_probabilities,
        share_rri,
    )

    print("\nAll explainability analyses completed.")


if __name__ == "__main__":
    main()
