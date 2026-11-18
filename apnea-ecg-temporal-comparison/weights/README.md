# Dataset

This repository does **not** include the data. The PhysioNet Apnea-ECG Database is public:

https://physionet.org/content/apnea-ecg/1.0.0/

Reference: Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The Apnea-ECG Database.
Computers in Cardiology 2000;27:255-258.

## What to put in this folder

Download the database and copy its files here, unchanged:

- `a01`-`a20`, `b01`-`b05`, `c01`-`c10`: training recordings (`.dat`, `.hea`, `.apn`)
- `x01`-`x35`: test recordings (`.dat`, `.hea`)
- `event-2-answers`: test-set labels
- `additional-information.txt`: used for the subject-level evaluation

## Preprocessed file (created by the code)

```
cd src
python preprocess_apnea_ecg.py
```

This creates `apnea-ecg.pkl` in this folder. All training and evaluation scripts read that file.

To use a different location, set the environment variable `APNEA_DATA_DIR`.
