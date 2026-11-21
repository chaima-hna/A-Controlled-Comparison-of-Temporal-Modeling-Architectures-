# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
from torch_config import *
from torch_data_loading import scaler, ApneaDataset, load_data
from model_mamba import DualBranchMamba, count_params

# =========================
# LOAD DATA
# =========================
x_train, y_train, x_val, y_val, x_test, y_test = load_data()
print(f"Train:{x_train.shape}  Val:{x_val.shape}  Test:{x_test.shape}")

train_loader = DataLoader(ApneaDataset(x_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(ApneaDataset(x_val, y_val),
                          batch_size=512, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(ApneaDataset(x_test, y_test),
                          batch_size=512, shuffle=False,
                          num_workers=2, pin_memory=True)


# =========================
# INTRA-CLASS MIXUP
# =========================

def mixup_batch(x, y_onehot, alpha=0.2):
    """50% intra-class + 50% cross-class MixUp."""
    n      = x.size(0)
    lam    = np.random.beta(alpha, alpha)
    labels = y_onehot.argmax(dim=1).cpu().numpy()

    idx_intra = np.arange(n)
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > 1:
            idx_intra[cls_idx] = cls_idx[np.random.permutation(len(cls_idx))]

    idx_cross = np.random.permutation(n)
    use_intra = np.random.rand(n) < 0.5
    idx_final = torch.tensor(
        np.where(use_intra, idx_intra, idx_cross),
        dtype=torch.long, device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[idx_final]
    y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[idx_final]
    return x_mix, y_mix


# =========================
# FOCAL LOSS
# =========================

class FocalLoss(nn.Module):

    def __init__(self, y_train, gamma=2.0):
        super().__init__()
        n_pos     = float((y_train == 1).sum())
        n_neg     = float((y_train == 0).sum())
        total     = n_pos + n_neg
        alpha_pos = total / (2.0 * n_pos)
        alpha_neg = total / (2.0 * n_neg)
        self.alpha = torch.tensor([alpha_neg, alpha_pos],
                                  dtype=torch.float32)
        self.gamma = gamma
        print(f"Focal loss -> alpha_normal={alpha_neg:.3f}  "
              f"alpha_apnea={alpha_pos:.3f}  gamma={gamma}")

    def forward(self, logits, y_onehot):
        probs   = F.softmax(logits, dim=-1).clamp(1e-7, 1.0 - 1e-7)
        p_t     = (probs * y_onehot).sum(dim=-1)
        alpha_t = (self.alpha.to(logits.device) * y_onehot).sum(dim=-1)
        focal_w = (1.0 - p_t) ** self.gamma
        return (-alpha_t * focal_w * torch.log(p_t)).mean()


# =========================
# COSINE LR WITH WARMUP
# =========================

def get_cosine_schedule(optimizer, warmup_steps, total_steps,
                        lr_min=1e-5, lr_max=1e-3):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min / lr_max + (1.0 - lr_min / lr_max) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================
# PREDICTION (single pass, no TTA)
# =========================

def predict_probs(model, x_np, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x_np), batch_size):
            x_t    = torch.tensor(x_np[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(x_t)
            out.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


# =========================
# THRESHOLD SEARCH
# =========================

def find_threshold(val_probs, y_val_true):
    best_t = 0.5; best_f1 = 0.0
    for t in np.arange(0.25, 0.66, 0.01):
        preds = (val_probs[:, 1] >= t).astype(int)
        f1    = f1_score(y_val_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    print(f"  Best threshold: {best_t:.2f}  (val_f1={best_f1:.4f})")
    return best_t


# =========================
# BUILD MODEL
# =========================

model        = DualBranchMamba().to(DEVICE)
total_params = count_params(model)


# =========================
# LOSS + OPTIMIZER + SCHEDULER
# =========================

criterion    = FocalLoss(y_train, gamma=2.0).to(DEVICE)
optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3)

n_steps      = len(train_loader) * EPOCHS
warmup_steps = len(train_loader) * 5
scheduler    = get_cosine_schedule(optimizer, warmup_steps, n_steps)

savepath      = os.path.join(weights_dir, "cnn_mamba.weights.pt")
savepath_full = os.path.join(weights_dir, "cnn_mamba.checkpoint.pt")


# =========================
# TRAINING LOOP
# =========================

best_val_f1 = 0.0
best_epoch  = 0
history     = {"loss":[], "acc":[], "val_loss":[], "val_acc":[], "val_f1":[]}

for epoch in range(EPOCHS):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    tl = ta = nb = 0

    for x_b, y_b in train_loader:
        x_b      = x_b.to(DEVICE)
        y_onehot = F.one_hot(y_b, num_classes=2).float().to(DEVICE)

        x_mix, y_mix = mixup_batch(x_b, y_onehot)

        optimizer.zero_grad()
        logits = model(x_mix)
        loss   = criterion(logits, y_mix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        acc = (logits.argmax(1) == y_mix.argmax(1)).float().mean()
        tl += loss.item(); ta += acc.item(); nb += 1

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    vl = va = nv = 0
    yt = []; yp_all = []

    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b      = x_b.to(DEVICE)
            y_onehot = F.one_hot(y_b, num_classes=2).float().to(DEVICE)
            logits   = model(x_b)
            loss     = criterion(logits, y_onehot)
            acc      = (logits.argmax(1) == y_b.to(DEVICE)).float().mean()
            vl += loss.item(); va += acc.item(); nv += 1
            yt.extend(y_b.numpy())
            yp_all.extend(logits.argmax(1).cpu().numpy())

    v_f1   = f1_score(yt, yp_all, zero_division=0)
    cur_lr = scheduler.get_last_lr()[0]

    print(f"Ep {epoch+1:3d}/{EPOCHS}  lr={cur_lr:.2e}"
          f"  tr={tl/nb:.4f}/{ta/nb:.4f}"
          f"  val={vl/nv:.4f}/{va/nv:.4f}  f1={v_f1:.4f}")

    history["loss"].append(tl/nb);     history["acc"].append(ta/nb)
    history["val_loss"].append(vl/nv); history["val_acc"].append(va/nv)
    history["val_f1"].append(v_f1)

    if v_f1 > best_val_f1:
        best_val_f1 = v_f1; best_epoch = epoch + 1

        # Save state dict
        torch.save(model.state_dict(), savepath)

        # Save full checkpoint
        torch.save({
            "epoch":        epoch + 1,
            "model_name":   "cnn_mamba",
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "val_f1":       v_f1,
            "val_acc":      va / nv,
            "history":      history,
            "total_params": total_params,
        }, savepath_full)

        print(f"  saved  f1={v_f1:.4f}")
        print(f"         {savepath}")
        print(f"         {savepath_full}")

print(f"\nBest val_f1={best_val_f1:.4f} at epoch {best_epoch}")


# =========================
# TRAINING PLOTS
# =========================

epochs_range = range(1, EPOCHS + 1)
fig, axes    = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Training History - M7 (Official Mamba)",
             fontsize=13, fontweight="bold")

axes[0].plot(epochs_range, history["loss"],     label="Train", color="steelblue")
axes[0].plot(epochs_range, history["val_loss"], label="Val",   color="coral", ls="--")
axes[0].axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"Best ep {best_epoch}")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["acc"],     label="Train", color="steelblue")
axes[1].plot(epochs_range, history["val_acc"], label="Val",   color="coral", ls="--")
axes[1].axvline(best_epoch, color="green", ls=":", lw=1.5)
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs_range, history["val_f1"], label="Val F1", color="mediumseagreen")
axes[2].axvline(best_epoch, color="green", ls=":", lw=1.5)
axes[2].set_title("Val F1"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "history_cnn_mamba.png"),
            dpi=600, bbox_inches="tight")
plt.show()


# =========================
# FINAL EVALUATION
# =========================

# Load best checkpoint
best_model = DualBranchMamba().to(DEVICE)
best_model.load_state_dict(torch.load(savepath, map_location=DEVICE))
best_model.eval()

# Print checkpoint metadata
ckpt = torch.load(savepath_full, map_location=DEVICE)
print(f"\nLoaded checkpoint:")
print(f"  Model      : {ckpt['model_name']}")
print(f"  Best epoch : {ckpt['epoch']}")
print(f"  Val F1     : {ckpt['val_f1']:.4f}")
print(f"  Val ACC    : {ckpt['val_acc']:.4f}")
print(f"  Params     : {ckpt['total_params']:,}")

# Predictions (single pass, no TTA)
print("\nPredicting on val set...")
val_probs  = predict_probs(best_model, x_val)

print("Predicting on test set...")
test_probs = predict_probs(best_model, x_test)

# Save test probabilities for bootstrap_ci.py
pred_dir = os.path.join(out_dir, "predictions")
os.makedirs(pred_dir, exist_ok=True)
np.savez(os.path.join(pred_dir, "cnn_mamba.npz"),
         y_true=np.asarray(y_test), prob=test_probs[:, 1])
print(f"Saved test predictions to {os.path.join(pred_dir, 'cnn_mamba.npz')}")

# Threshold search on val
best_t = find_threshold(val_probs, y_val)

# Final metrics on test
y_pred = (test_probs[:, 1] >= best_t).astype(int)
roc    = roc_auc_score(np.eye(2)[y_test], test_probs)
C      = confusion_matrix(y_test, y_pred, labels=(1, 0))
TP, TN, FP, FN = C[0,0], C[1,1], C[1,0], C[0,1]

acc = (TP + TN) / (TP + TN + FP + FN)
sn  =  TP / (TP + FN) if (TP + FN) > 0 else 0.0
sp  =  TN / (TN + FP) if (TN + FP) > 0 else 0.0
f1  = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  M7 OFFICIAL MAMBA — FINAL RESULTS")
print(f"{'='*50}")
print(f"  ACC    : {acc:.4f}")
print(f"  SN     : {sn:.4f}")
print(f"  SP     : {sp:.4f}")
print(f"  F1     : {f1:.4f}")
print(f"  AUC    : {roc:.4f}")
print(f"  Thr    : {best_t:.2f}")
print(f"  Params : {total_params:,}")
print(f"{'='*50}")
