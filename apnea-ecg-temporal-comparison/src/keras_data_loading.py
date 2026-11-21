from keras_config import *

# ── Normalize ─────────────────────────────────────────────────────────────────

def scaler(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(PKL_PATH, "rb") as f:
        apnea_ecg = pickle.load(f)

    x = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train     = apnea_ecg["groups_train"]

    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
        rri_interp  = splev(tm, splrep(rri_tm,  scaler(rri_signal),  k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x.append([rri_interp, ampl_interp])

    num = list(range(len(x)))
    if os.environ.get("APNEA_VAL_SPLIT", "segment") == "recording":
        # Robustness check: whole recordings are held out for validation, so no
        # overlapping window is shared between training and validation.
        # c05 and c06 come from the same original recording and are kept together.
        from sklearn.model_selection import GroupShuffleSplit
        groups = np.array(["c05" if g == "c06" else g for g in groups_train])
        trainlist, vallist = next(GroupShuffleSplit(
            n_splits=1, test_size=0.3, random_state=42).split(num, y_train, groups))
        y_all = np.array(y_train)
        y_tr, y_vl = y_all[trainlist], y_all[vallist]
        print("Validation recordings:", sorted({str(g) for g in np.array(groups_train)[vallist]}))
    else:
        # Protocol of the paper: random 70/30 split of segments, stratified by class.
        trainlist, vallist, y_tr, y_vl = train_test_split(
            num, y_train, test_size=0.3, random_state=42, stratify=y_train)

    x_train, x_val = [], []
    for i in trainlist: x_train.append(x[i])
    for i in vallist:   x_val.append(x[i])

    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    x_val   = np.array(x_val,   dtype="float32").transpose((0, 2, 1))

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
        rri_interp  = splev(tm, splrep(rri_tm,  scaler(rri_signal),  k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test.append([rri_interp, ampl_interp])

    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))

    return x_train, np.array(y_tr), x_val, np.array(y_vl), x_test, np.array(y_test)
