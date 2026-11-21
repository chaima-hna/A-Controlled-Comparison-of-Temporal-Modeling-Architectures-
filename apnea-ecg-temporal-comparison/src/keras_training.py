# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
from keras_config import *
from keras_base_model import CUSTOM_OBJ

# ── MixUp ─────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, alpha=0.2):
     
    n      = len(x)
    lam    = np.random.beta(alpha, alpha)
    labels = np.argmax(y, axis=1)

    idx_intra = np.arange(n)
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > 1:
            idx_intra[cls_idx] = cls_idx[np.random.permutation(len(cls_idx))]

    idx_cross = np.random.permutation(n)
    use_intra = np.random.rand(n) < 0.5
    idx_final = np.where(use_intra, idx_intra, idx_cross)

    x_mix = (lam * x + (1.0 - lam) * x[idx_final]).astype("float32")
    y_mix = (lam * y + (1.0 - lam) * y[idx_final]).astype("float32")
    return x_mix, y_mix


def make_focal_loss(y_train_cat, gamma=2.0):
    n_pos    = float(np.sum(y_train_cat[:, 1]))
    n_neg    = float(np.sum(y_train_cat[:, 0]))
    total    = n_pos + n_neg
    alpha_pos = total / (2.0 * n_pos)
    alpha_neg = total / (2.0 * n_neg)

    def compute_loss(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred_c = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t      = tf.reduce_sum(y_true * y_pred_c, axis=-1)
        is_apnea = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
        alpha_t  = is_apnea * alpha_pos + (1.0 - is_apnea) * alpha_neg
        focal_w  = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(-alpha_t * focal_w * tf.math.log(p_t))

    return compute_loss

# ── Cosine LR with warmup ─────────────────────────────────────────────────────

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_max, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_max       = float(lr_max)
        self.lr_min       = float(lr_min)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.lr_max * (step / self.warmup_steps)
        cosine = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                 1.0 + tf.cos(np.pi * (step - self.warmup_steps) /
                               (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"lr_max": self.lr_max, "lr_min": self.lr_min,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

 

def find_threshold(val_probs, y_val):
    best_t = 0.5; best_f1 = 0.0
    for t in np.arange(0.25, 0.66, 0.01):
        preds = (val_probs[:, 1] >= t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return best_t, best_f1

 

def run_training(model, model_name, x_train, y_train_cat,
                 x_val, y_val, y_val_cat,
                 x_test, y_test, y_test_cat):

    tf.random.set_seed(SEED); np.random.seed(SEED)

    N_TR         = len(x_train)
    N_VL         = len(x_val)
    n_steps      = (N_TR // BATCH_SIZE) * EPOCHS
    warmup_steps = (N_TR // BATCH_SIZE) * 5

    compute_loss = make_focal_loss(y_train_cat)
    schedule     = WarmupCosineSchedule(1e-3, 1e-5, warmup_steps, n_steps)
    optimizer    = tf.keras.optimizers.Adam(learning_rate=schedule)
    savepath     = os.path.join(weights_dir, f"{model_name}.weights.h5")

    model.summary(line_length=70)

    # tf.function steps defined here — captures this model/optimizer only
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss   = compute_loss(y_batch, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1)), tf.float32))
        return loss, acc

    @tf.function
    def val_step(x_batch, y_batch):
        y_pred = model(x_batch, training=False)
        loss   = compute_loss(y_batch, y_pred)
        acc    = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1)), tf.float32))
        return loss, acc

    best_f1 = 0.0; best_ep = 0
    history = {"loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[], "val_f1":[]}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        idx = np.random.permutation(N_TR)
        tl = ta = nb = 0
        for s in range(0, N_TR, BATCH_SIZE):
            b            = idx[s:s + BATCH_SIZE]
            x_mix, y_mix = mixup_batch(x_train[b], y_train_cat[b])
            l, a         = train_step(tf.constant(x_mix), tf.constant(y_mix))
            tl += float(l); ta += float(a); nb += 1

        vl = va = nv = 0; yt = []; yp_all = []
        for s in range(0, N_VL, BATCH_SIZE):
            xb = x_val[s:s+BATCH_SIZE]; yb = y_val_cat[s:s+BATCH_SIZE]
            l, a = val_step(xb, yb)
            vl += float(l); va += float(a); nv += 1
            yp = model(xb, training=False).numpy()
            yt.extend(np.argmax(yb, -1)); yp_all.extend(np.argmax(yp, -1))

        v_f1 = f1_score(yt, yp_all, zero_division=0)
        print(f"  tr={tl/nb:.4f}/{ta/nb:.4f}  val={vl/nv:.4f}/{va/nv:.4f}  f1={v_f1:.4f}")

        history["loss"].append(tl/nb); history["accuracy"].append(ta/nb)
        history["val_loss"].append(vl/nv); history["val_accuracy"].append(va/nv)
        history["val_f1"].append(v_f1)

        if v_f1 > best_f1:
            best_f1 = v_f1; best_ep = epoch + 1
            model.save_weights(savepath)
            print(f"  ✓ saved  f1={v_f1:.4f}")

    print(f"\nBest val_f1={best_f1:.4f} at epoch {best_ep}")

    # ── Training plot ─────────────────────────────────────────────────────────
    epochs_range = range(1, EPOCHS + 1)
    fig, axes    = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training — {model_name}", fontsize=18, fontweight="bold")

    axes[0].plot(epochs_range, history["loss"],         label="Train", color="steelblue")
    axes[0].plot(epochs_range, history["val_loss"],     label="Val",   color="coral", ls="--")
    axes[0].axvline(best_ep, color="green", ls=":", lw=1.5, label=f"Best ep {best_ep}")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["accuracy"],     label="Train", color="steelblue")
    axes[1].plot(epochs_range, history["val_accuracy"], label="Val",   color="coral", ls="--")
    axes[1].axvline(best_ep, color="green", ls=":", lw=1.5)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_range, history["val_f1"], label="Val F1", color="mediumseagreen")
    axes[2].axvline(best_ep, color="green", ls=":", lw=1.5)
    axes[2].set_title("Val F1"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"history_{model_name}.png"),
                dpi=600, bbox_inches="tight")
    plt.show()

    # ── Threshold + final eval (no TTA) ───────────────────────────────────────
    model.load_weights(savepath)   # restore the best-epoch weights
    best_model = model

    print("\nPredicting on val set...")
    val_probs  = best_model.predict(x_val,  batch_size=512, verbose=0)
    print("Predicting on test set...")
    test_probs = best_model.predict(x_test, batch_size=512, verbose=0)

    best_t, _ = find_threshold(val_probs, y_val)
    print(f"Optimal threshold: {best_t:.2f}")

    y_pred = (test_probs[:, 1] >= best_t).astype(int)
    roc    = roc_auc_score(y_test_cat, test_probs)
    C      = confusion_matrix(y_test, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0,0], C[1,1], C[1,0], C[0,1]

    acc = (TP + TN) / (TP + TN + FP + FN)
    sn  =  TP / (TP + FN) if (TP + FN) > 0 else 0.0
    sp  =  TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1  = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  ACC  : {acc:.4f}")
    print(f"  SN   : {sn:.4f}")
    print(f"  SP   : {sp:.4f}")
    print(f"  F1   : {f1:.4f}")
    print(f"  AUC  : {roc:.4f}")
    print(f"  Thr  : {best_t:.2f}")
    print(f"  Params: {best_model.count_params():,}")
    print(f"{'='*50}")

    return {"name": model_name, "acc": acc, "sn": sn, "sp": sp,
            "f1": f1, "auc": roc, "threshold": best_t,
            "params": best_model.count_params(), "best_epoch": best_ep}
