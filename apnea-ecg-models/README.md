# Temporal Modeling Architectures for ECG-Based Obstructive Sleep Apnea Detection

Code and results for the paper **"A Controlled Comparison of Temporal Modeling Architectures for ECG-Based Obstructive Sleep Apnea Detection with Integrated Gradients Analysis"**.

## 1. What is this work? (Definition)

Obstructive Sleep Apnea (OSA) is a sleep disorder where breathing stops repeatedly during sleep. It can cause serious heart problems. The standard test (polysomnography) is expensive and needs a sleep lab. A single-lead ECG is a cheap signal that can be used for automatic screening.

This work builds a deep learning model that detects OSA **minute by minute** from a single-lead ECG. The main goal is a **controlled comparison**: the same CNN backbone and the same training protocol are used, and only one part changes, the **temporal modeling module**. Six modules are compared:

- Transformer
- Mamba (selective state-space model)
- LSTM
- GRU
- BiLSTM
- BiGRU

An **Integrated Gradients** analysis is also done to see which input time steps and which channel (RRI or R-amplitude) the model uses.

## 2. What was done (step by step)

1. **Dataset.** PhysioNet Apnea-ECG v1.0.0: 70 overnight recordings (35 for learning, 35 for testing), 100 Hz, with one apnea label per minute.
2. **Preprocessing.**
   - Each minute is taken with 2 minutes before and 2 minutes after, giving a 5-minute segment.
   - Band-pass filter 3-45 Hz.
   - R-peak detection (Hamilton algorithm) and correction.
   - Two signals are extracted: **RRI** (R-R interval) and **R-peak amplitude**.
   - Cubic spline interpolation to 3 Hz, so each channel has 900 samples.
   - Min-max normalization.
3. **Model (shared backbone).**
   - Dual-branch CNN: one branch for RRI and one for R-amplitude (3 conv blocks each, kernel size 11).
   - The two outputs are concatenated into a (30, 64) feature sequence.
   - Efficient Channel Attention (ECA) block.
   - **Temporal module** (the part that changes): Transformer, Mamba, LSTM, GRU, BiLSTM or BiGRU.
   - Global average pooling + global max pooling, Dense 64, Softmax (Apnea / Normal).
4. **Training (identical for all 6 models).**
   - Class-weighted focal loss (gamma = 2.0).
   - MixUp (50% within-class, 50% between-class, alpha = 0.2).
   - Adam, warm-up (5 epochs) + cosine learning-rate decay, 120 epochs, batch size 128.
   - Gradient clipping (max norm 1.0), random seed 42.
   - Best checkpoint chosen by validation F1-score.
   - Test-time augmentation (10 noisy passes) and decision threshold chosen on the validation set.
5. **Evaluation.**
   - Segment level: ACC, SN, SP, F1, AUC with 95% bootstrap confidence intervals (1,000 resamples).
   - Record level: apnea burden (apnea minutes per hour), classification with a cutoff of 5, and mean absolute error (MAE).
6. **Ablation study.** Components are added one by one (ECA, GRU, warm-up cosine, MixUp, focal loss) to see the effect of each.
7. **Explainability.** Integrated Gradients (zero baseline, 50 steps) on the test set.

## 3. Results

### Segment level (16,945 test segments)

| Model | Params | ACC (%) | SN (%) | SP (%) | F1 (%) | AUC (%) | Threshold |
|---|---|---|---|---|---|---|---|
| CNN+Transformer | ~104K | 90.64 | 87.72 | 92.45 | 87.77 | 96.54 | 0.50 |
| CNN+Mamba | 101,973 | 90.39 | 86.59 | 92.75 | 87.35 | 96.26 | 0.54 |
| CNN+LSTM | ~101K | 90.82 | 85.10 | 94.37 | 87.65 | 96.57 | 0.60 |
| **CNN+GRU** | **84,725** | **90.99** | 87.89 | 92.92 | **88.20** | **96.81** | 0.50 |
| CNN+BiGRU | ~168K | 90.50 | 89.31 | 91.25 | 87.81 | 96.70 | 0.47 |
| CNN+BiLSTM | ~208K | 90.65 | 85.99 | 93.54 | 87.57 | 96.59 | 0.54 |

Full table with confidence intervals: [`results/table2_segment_metrics.csv`](results/table2_segment_metrics.csv)

### Record level (CNN+GRU, 35 test recordings)

| ACC | SN | SP | MAE |
|---|---|---|---|
| 97.14% | 100% | 91.67% | 2.82 apnea minutes per hour |

### Main findings

- The **GRU** gave the highest accuracy, F1-score and AUC, with the **fewest parameters** (84,725).
- The differences between the 6 models are **small** and the confidence intervals overlap. The results should not be read as one model being clearly better.
- The differences in sensitivity and specificity mostly come from the chosen threshold, not from the temporal module.
- The ablation study shows that adding a GRU layer to the CNN+ECA model gave the largest improvement.
- Integrated Gradients showed moderately concentrated attributions and only weak alignment with local changes in the signal.

## 4. Limitations

- One public benchmark only (70 recordings), no external validation.
- One training seed, so training variability is not measured.
- Validation F1 (0.944-0.950) is higher than test F1 (0.874-0.882). Overlapping windows may have been shared between the training and validation sets, so the split should be checked and repeated by recording.
- Mamba is implemented in PyTorch and the other five models in Keras. Parameter counts are computed differently (the Keras counts include 288 non-trainable batch-norm statistics).
- Inference time, memory and energy were not measured.
- Integrated Gradients results depend on the zero baseline.

## 5. Data

The raw data are **not** included in this repository. Download Apnea-ECG from PhysioNet:
<https://physionet.org/content/apnea-ecg/1.0.0/>

 
## 7. Citation

To be added after publication.
