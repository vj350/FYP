import time
import numpy as np

from scipy.signal import cheby2, filtfilt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)


def cheby2_bandpass_filter_epochs(X, lowcut, highcut, fs=250, order=6, rs=20):
    """
    Apply Chebyshev Type II band-pass filtering to epoched EEG.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_channels, n_times)
    lowcut : float
    highcut : float
    fs : int
    order : int
    rs : float
        Stopband attenuation in dB

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_channels, n_times)
    """
    nyq = 0.5 * fs
    b, a = cheby2(order, rs, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return filtfilt(b, a, X, axis=2)


def get_filter_bands(config: PreprocessingConfig):
    """
    FBCSP filter bank controlled by C and D.

    C1 = no theta
    C2 = include theta (4-8 Hz)

    D1 = up to 30 Hz
    D2 = up to 40 Hz
    """
    bands = []

    if config.C == 2:
        bands.append((4, 8))

    if config.D == 1:
        bands.extend([
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 24),
            (24, 28),
            (28, 30),
        ])
    elif config.D == 2:
        bands.extend([
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 24),
            (24, 28),
            (28, 32),
            (32, 36),
            (36, 40),
        ])
    else:
        raise ValueError("config.D must be 1 or 2")

    return bands


def select_top_mibif_features(X_train, y_train, X_test, k_features):
    """
    MIBIF-style feature selection using mutual information.

    Feature ranking is done on training data only.
    """
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    top_idx = np.argsort(mi_scores)[::-1][:k_features]

    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    return X_train_sel, X_test_sel, top_idx


def baseline_correct(X, fs=250, baseline_duration=0.5):
    """
    Subtract per-trial per-channel mean of first baseline_duration seconds.
    Supervisor paper Section II.A: first 0.5 s used as reference baseline.
    X shape: (N, T, C)
    """
    n_baseline = int(baseline_duration * fs)
    baseline_mean = X[:, :n_baseline, :].mean(axis=1, keepdims=True)
    return X - baseline_mean


def sliding_window_augment(X, y, window_samples=400, step_samples=100):
    """
    Augment training data by sliding a fixed-length window across each trial.

    Supervisor paper (Section II.D): 1.6 s window, 100-sample stride -> ~7x augmentation.

    Parameters
    ----------
    X             : np.ndarray, shape (N, T, C)  -- at 250 Hz
    y             : np.ndarray, shape (N,)
    window_samples: int  -- window length (400 = 1.6 s at 250 Hz)
    step_samples  : int  -- step size   (100 = 0.4 s at 250 Hz)

    Returns
    -------
    X_aug : np.ndarray, shape (N * n_windows, window_samples, C)
    y_aug : np.ndarray, shape (N * n_windows,)

    With a 1000-sample (4 s at 250 Hz) input and window=400, step=100:
        n_windows = (1000 - 400) // 100 + 1 = 7 windows per trial (~7x augmentation)
    """
    T = X.shape[1]
    X_aug, y_aug = [], []
    for i in range(X.shape[0]):
        for start in range(0, T - window_samples + 1, step_samples):
            X_aug.append(X[i, start:start + window_samples, :])
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)


def run_csp_svm_holdout(X_train, y_train, X_test, y_test, n_csp_components=2,
                        fs=250, augment=True):
    """
    Train on X_train, evaluate on X_test.

    Preprocessing:
    - augment=True:  sliding window augmentation (~11x) on train, single crop on test
    - augment=False: single [0.5, 2.5]s crop for both train and test
    """
    t_start = int(0.5 * fs)          # 125
    t_end   = t_start + 400          # 525  (1.6 s window)

    # Baseline correction (supervisor paper Section II.A)
    X_train = baseline_correct(X_train, fs=fs)
    X_test  = baseline_correct(X_test,  fs=fs)

    if augment:
        X_train, y_train = sliding_window_augment(X_train, y_train)
    else:
        X_train = X_train[:, t_start:t_end, :]

    X_test = X_test[:, t_start:t_end, :]

    # (N, T, C) -> (N, C, T) for MNE CSP
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test  = np.transpose(X_test,  (0, 2, 1))

    csp = CSP(n_components=n_csp_components, log=True, norm_trace=False)
    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat  = csp.transform(X_test)

    clf = SVC(kernel="linear")
    start = time.time()
    clf.fit(X_train_feat, y_train)
    end = time.time()

    y_pred = clf.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"CSP accuracy: {acc:.4f}  kappa: {kappa:.2f}  time: {end-start:.1f}s")
    return acc, kappa, end - start


def run_fbcsp_svm_holdout(X_train, y_train, X_test, y_test, config,
                          n_csp_components=2, fs=250, k_features=8, augment=True):
    """
    Train on X_train, evaluate on X_test.

    Preprocessing (per supervisor's BF pipeline for FBCSP):
    - augment=True:  sliding window augmentation (~11x) on train, single crop on test
    - augment=False: single [0.5, 2.5]s crop for both train and test
    - FBCSP filter bank applied per band (Chebyshev Type II)
    - MIBIF feature selection
    - Linear SVM classifier
    """
    t_start = int(0.5 * fs)          # 125
    t_end   = t_start + 400          # 525  (1.6 s window)

    # Baseline correction (supervisor paper Section II.A)
    X_train = baseline_correct(X_train, fs=fs)
    X_test  = baseline_correct(X_test,  fs=fs)

    if augment:
        X_train, y_train = sliding_window_augment(X_train, y_train)
    else:
        X_train = X_train[:, t_start:t_end, :]

    X_test = X_test[:, t_start:t_end, :]

    # (N, T, C) -> (N, C, T) for filtering
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test  = np.transpose(X_test,  (0, 2, 1))

    bands = get_filter_bands(config)
    train_feats, test_feats = [], []

    for lowcut, highcut in bands:
        X_tr_band = cheby2_bandpass_filter_epochs(X_train, lowcut, highcut, fs=fs)
        X_te_band = cheby2_bandpass_filter_epochs(X_test,  lowcut, highcut, fs=fs)

        csp = CSP(n_components=n_csp_components, log=True, norm_trace=False)
        train_feats.append(csp.fit_transform(X_tr_band, y_train))
        test_feats.append(csp.transform(X_te_band))

    X_train_all = np.concatenate(train_feats, axis=1)
    X_test_all  = np.concatenate(test_feats,  axis=1)

    k_use = min(k_features, X_train_all.shape[1])
    X_train_sel, X_test_sel, _ = select_top_mibif_features(
        X_train_all, y_train, X_test_all, k_use)

    clf = SVC(kernel="linear")
    start = time.time()
    clf.fit(X_train_sel, y_train)
    end = time.time()

    y_pred = clf.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"FBCSP accuracy: {acc:.4f}  kappa: {kappa:.2f}  time: {end-start:.1f}s")
    return acc, kappa, end - start


if __name__ == "__main__":
    files = get_training_files("data/2b")

    # Change this for experiments
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running experiment with config:", config)

    # Start with one subject first
    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups is None:", groups is None)
    if groups is not None:
        print("groups shape:", groups.shape)

    print("\nUse run_all.py to run experiments.")
