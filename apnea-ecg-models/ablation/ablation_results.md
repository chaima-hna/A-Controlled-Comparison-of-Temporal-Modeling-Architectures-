# Ablation study – progressive (each step adds one component to the previous one)

All steps: PhysioNet Apnea-ECG, official test set (x01–x35), per-minute classification, 120 epochs, Adam, best epoch by validation F1.
Run each file on its own (`python ablation_1_cnn.py`, ...). Values below are copied from the notebook outputs.

| Step | File | Model | ACC | SN | SP | F1 | AUC |
|---|---|---|---|---|---|---|---|
| 1 | `ablation_1_cnn.py` | dual-branch CNN (RRI + amplitude), GAP+GMP | 0.8719 | 0.8424 | 0.8903 | 0.8344 | 0.9440 |
| 2 | `ablation_2_cnn_eca.py` | + ECA channel attention | 0.8786 | 0.8123 | 0.9198 | 0.8368 | 0.9455 |
| 3 | `ablation_3_cnn_eca_gru.py` | + 2 × GRU(64) | 0.8945 | 0.8490 | 0.9228 | 0.8605 | 0.9566 |
| 4 | `ablation_4_cnn_eca_gru_cosine_lr.py` | + warm-up cosine learning rate | 0.8981 | 0.8565 | 0.9240 | 0.8656 | 0.9556 |
| 5 | `ablation_5_cnn_eca_gru_cosine_lr_mixup.py` | + MixUp | 0.9048 | 0.8857 | 0.9167 | 0.8770 | 0.9533 |
| final | `keras_models/model_gru.py` | + focal loss, 10× TTA, threshold search | 0.9099 | 0.8789 | 0.9292 | 0.8820 | 0.9681 |
