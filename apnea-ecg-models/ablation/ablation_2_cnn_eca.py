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

# ───────────────────────── CONFIG ─────────────────────────
base_dir = "/kaggle/input/datasets/chaimahannachi/apneaecg"
save_path = "/kaggle/working/cnn_eca_model.keras"

ir, before, after = 3, 2, 2
BATCH_SIZE = 128
EPOCHS = 120
SEED = 42

# ───────────────────────── NORMALIZATION ─────────────────────────
def scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

# ───────────────────────── LOAD DATA ─────────────────────────
def load_data():
    path = "apnea-ecg (1).pkl"
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, path), "rb") as f:
        data = pickle.load(f)

    x, y_train = [], data["y_train"]

    o_train = data["o_train"]
    o_test  = data["o_test"]
    y_test  = data["y_test"]

    for i in range(len(o_train)):
        (rri_tm, rri_sig), (amp_tm, amp_sig) = o_train[i]

        rri = splev(tm, splrep(rri_tm, scaler(rri_sig), k=3), ext=1)
        amp = splev(tm, splrep(amp_tm, scaler(amp_sig), k=3), ext=1)

        x.append([rri, amp])

    x = np.asarray(x, dtype=np.float32).transpose((0, 2, 1))
    y_train = np.asarray(y_train)

    idx = np.arange(len(x))
    tr, vl = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y_train)

    x_train, x_val = x[tr], x[vl]
    y_train, y_val = y_train[tr], y_train[vl]

    # TEST
    x_test = []
    for i in range(len(o_test)):
        (rri_tm, rri_sig), (amp_tm, amp_sig) = o_test[i]

        rri = splev(tm, splrep(rri_tm, scaler(rri_sig), k=3), ext=1)
        amp = splev(tm, splrep(amp_tm, scaler(amp_sig), k=3), ext=1)

        x_test.append([rri, amp])

    x_test = np.asarray(x_test, dtype=np.float32).transpose((0, 2, 1))
    y_test = np.asarray(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

# ───────────────────────── ECA BLOCK ─────────────────────────
@tf.keras.utils.register_keras_serializable()
class ECABlock(tf.keras.layers.Layer):

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(1, kernel_size, padding="same", use_bias=False)

    def call(self, x):
        gap = tf.reduce_mean(x, axis=1, keepdims=False)
        gap = tf.expand_dims(gap, axis=-1)

        weights = tf.nn.sigmoid(self.conv(gap))
        weights = tf.expand_dims(tf.squeeze(weights, axis=-1), axis=1)

        return x * weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg

# ───────────────────────── CNN BRANCH ─────────────────────────
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

# ───────────────────────── MODEL ─────────────────────────
def create_model(input_shape):
    inp = Input(shape=input_shape)

    # Clean native slicing syntax (Removes Lambda serialization errors)
    rri = inp[:, :, 0:1]
    amp = inp[:, :, 1:2]

    f1 = branch(rri)
    f2 = branch(amp)

    fused = Concatenate(axis=-1)([f1, f2])

    # ✔ ECA added here
    fused = ECABlock()(fused)

    gap = GlobalAveragePooling1D()(fused)
    gmp = GlobalMaxPooling1D()(fused)

    # Concat gap and gmp cleanly
    x = Concatenate()([gap, gmp])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    out = Dense(2, activation="softmax")(x)

    return Model(inp, out)

# ───────────────────────── F1 & SAVE BEST CALLBACK ─────────────────────────
class F1CheckpointCallback(Callback):

    def __init__(self, validation_data, filepath):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.filepath = filepath
        self.best_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.predict(self.x_val, verbose=0)
        pred = np.argmax(pred, axis=1)
        f1 = f1_score(self.y_val, pred)
        logs["val_f1"] = f1

        print(f" - val_f1: {f1:.4f}")

        # Save model exclusively if validation F1 increases
        if f1 > self.best_f1:
            print(f" -> val_f1 improved from {self.best_f1:.4f} to {f1:.4f}. Saving model snapshot...")
            self.best_f1 = f1
            self.model.save(self.filepath)

# ───────────────────────── TRAIN ─────────────────────────
def run():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    print("Shapes:", x_train.shape, y_train.shape)

    model = create_model(x_train.shape[1:])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    f1_save_callback = F1CheckpointCallback(
        validation_data=(x_val, y_val),
        filepath=save_path
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[f1_save_callback]
    )

    # ───────── EVALUATION USING BEST SNAPSHOT ─────────
    print("\nLoading the best saved model checkpoint for testing...")
    best_model = tf.keras.models.load_model(
        save_path, 
        custom_objects={"ECABlock": ECABlock}
    )

    probs = best_model.predict(x_test)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=(1, 0))
    TP, TN, FP, FN = cm[0,0], cm[1,1], cm[1,0], cm[0,1]

    acc = (TP + TN) / (TP + TN + FP + FN)
    sn  = TP / (TP + FN + 1e-8)
    sp  = TN / (TN + FP + 1e-8)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs[:, 1])

    print("\n===== FINAL RESULTS (CNN + ECA - BEST MODEL) =====")
    print("ACC:", acc)
    print("SN :", sn)
    print("SP :", sp)
    print("F1 :", f1)
    print("AUC:", auc)

if __name__ == "__main__":
    run()
