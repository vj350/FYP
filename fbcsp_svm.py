import time
import numpy as np

from scipy.signal import cheby2, filtfilt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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


def run_csp_svm_holdout(X_train, y_train, X_test, y_test, n_csp_components=2):
    """Train on X_train, evaluate on X_test (T→E protocol)."""
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    csp = CSP(n_components=n_csp_components, log=True, norm_trace=False)
    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat = csp.transform(X_test)

    clf = SVC(kernel="linear")
    start = time.time()
    clf.fit(X_train_feat, y_train)
    end = time.time()

    y_pred = clf.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    print(f"CSP holdout accuracy: {acc:.4f}")
    return acc, end - start


def run_fbcsp_svm_holdout(X_train, y_train, X_test, y_test, config,
                          n_csp_components=2, fs=250, k_features=8):
    """Train on X_train, evaluate on X_test (T→E protocol)."""
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    bands = get_filter_bands(config)
    train_feats, test_feats = [], []

    for lowcut, highcut in bands:
        X_tr_band = cheby2_bandpass_filter_epochs(X_train, lowcut, highcut, fs=fs)
        X_te_band = cheby2_bandpass_filter_epochs(X_test, lowcut, highcut, fs=fs)

        csp = CSP(n_components=n_csp_components, log=True, norm_trace=False)
        train_feats.append(csp.fit_transform(X_tr_band, y_train))
        test_feats.append(csp.transform(X_te_band))

    X_train_all = np.concatenate(train_feats, axis=1)
    X_test_all = np.concatenate(test_feats, axis=1)

    k_use = min(k_features, X_train_all.shape[1])
    X_train_sel, X_test_sel, _ = select_top_mibif_features(X_train_all, y_train, X_test_all, k_use)

    clf = SVC(kernel="linear")
    start = time.time()
    clf.fit(X_train_sel, y_train)
    end = time.time()

    y_pred = clf.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    print(f"FBCSP holdout accuracy: {acc:.4f}")
    return acc, end - start


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

    print("\nUse run_all.py to run experiments with the T-E holdout protocol.")