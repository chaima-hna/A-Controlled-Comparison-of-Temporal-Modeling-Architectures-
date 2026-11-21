import os
import pickle
import numpy as np
import tensorflow as tf

from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.interpolate import splev, splrep

# ================= CONFIG =================
from paths import PKL_PATH, WEIGHTS_DIR
from keras_data_loading import load_data

save_path = str(WEIGHTS_DIR / "ablation_1_cnn.keras")

ir, before, after = 3, 2, 2

BATCH_SIZE = 128
EPOCHS = 120
SEED = 42
THRESHOLD = 0.49   # decision threshold on P(apnea)

# ───────────────────────── LOAD DATA ─────────────────────────
# Loaded with the shared loader (keras_data_loading.load_data).
# APNEA_VAL_SPLIT=recording holds out whole recordings for validation.


# ================= CNN BRANCH =================
def branch(x):
    x = Conv1D(16, 11, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(24, 11, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling1D(3, padding="same")(x)

    x = Conv1D(32, 11, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling1D(5, padding="same")(x)

    return x


# ================= MODEL =================
def create_model(input_shape):
    inp = Input(shape=input_shape)

    # Replaced Lambda layers with direct slicing to prevent serialization errors
    rri = inp[:, :, 0:1]
    amp = inp[:, :, 1:2]

    f1 = branch(rri)
    f2 = branch(amp)

    fused = Concatenate(axis=-1)([f1, f2])

    gap = GlobalAveragePooling1D()(fused)
    gmp = GlobalMaxPooling1D()(fused)

    x = Concatenate()([gap, gmp])

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)   # same dropout as the final CNN+GRU model

    out = Dense(2, activation="softmax")(x)

    return Model(inp, out)


# ================= LOSS =================
def get_loss(y):
    if len(y.shape) == 1:
        return tf.keras.losses.SparseCategoricalCrossentropy()

    return tf.keras.losses.CategoricalCrossentropy()


# ================= F1 & SAVE BEST CALLBACK =================
class F1CheckpointCallback(Callback):

    def __init__(self, validation_data, filepath):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.filepath = filepath
        self.best_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.predict(self.x_val, verbose=0)
        pred = (pred[:, 1] >= THRESHOLD).astype(int)
        f1 = f1_score(self.y_val, pred)
        logs["val_f1"] = f1

        print(f" - val_f1: {f1:.4f}")

        # Overwrite file only if the F1 score improves
        if f1 > self.best_f1:
            print(f" -> val_f1 improved from {self.best_f1:.4f} to {f1:.4f}. Saving model snapshot...")
            self.best_f1 = f1
            self.model.save(self.filepath)


# ================= TRAIN =================
def run():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    print("Training shape:", x_train.shape)

    model = create_model(x_train.shape[1:])
    loss_fn = get_loss(y_train)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # Setup the metric tracker and checkpoint saver
    f1_save_callback = F1CheckpointCallback(
        validation_data=(x_val, y_val),
        filepath=save_path
    )

    # Fits for exactly 120 epochs without an early stop threshold
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[f1_save_callback],
        verbose=1
    )

    # ================= TEST USING BEST MODEL =================
    print("\nLoading the best saved model from training for test evaluation...")
    best_model = tf.keras.models.load_model(save_path)

    probs = best_model.predict(x_test, verbose=0)
    y_pred = (probs[:, 1] >= THRESHOLD).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 0]
    )

    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN + 1e-8)
    sp = TN / (TN + FP + 1e-8)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs[:, 1])

    print("\n========== FINAL RESULTS (BEST MODEL) ==========")
    print("Accuracy :", acc)
    print("Sensitivity :", sn)
    print("Specificity :", sp)
    print("F1-score :", f1)
    print("AUC :", auc)


if __name__ == "__main__":
    run()
