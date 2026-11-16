# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
from torch_config import *

# =========================
# NORMALIZE
# =========================

def scaler(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# =========================
# DATASET
# =========================

class ApneaDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y.astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =========================
# LOAD DATA
# =========================

def load_data():
    path = "apnea-ecg (1).pkl"
    tm   = np.arange(0, (before + 1 + after) * 60, step=1.0 / ir)

    with open(os.path.join(base_dir, path), "rb") as f:
        apnea_ecg = pickle.load(f)

    x       = []
    o_train = apnea_ecg["o_train"]
    y_train = apnea_ecg["y_train"]

    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
        rri_interp  = splev(tm, splrep(rri_tm,  scaler(rri_signal),  k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x.append([rri_interp, ampl_interp])

    num = list(range(len(x)))
    tr_idx, vl_idx, y_tr, y_vl = train_test_split(
        num, y_train, test_size=0.3, random_state=42, stratify=y_train)

    x_train = np.array([x[i] for i in tr_idx],
                       dtype="float32").transpose(0, 2, 1)   # (N, 900, 2)
    x_val   = np.array([x[i] for i in vl_idx],
                       dtype="float32").transpose(0, 2, 1)

    x_test_raw = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
        rri_interp  = splev(tm, splrep(rri_tm,  scaler(rri_signal),  k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test_raw.append([rri_interp, ampl_interp])

    x_test = np.array(x_test_raw, dtype="float32").transpose(0, 2, 1)

    return (x_train, np.array(y_tr).astype(int),
            x_val,   np.array(y_vl).astype(int),
            x_test,  np.array(y_test).astype(int))
