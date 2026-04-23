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
from cross_validation import make_cv_splits


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


def run_csp_svm_cv(X, y, groups, config, n_splits=10, n_csp_components=2):
    """
    Run CSP + SVM baseline with cross-validation.
    """
    # MNE CSP expects (samples, channels, time)
    X = np.transpose(X, (0, 2, 1))

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running CSP + SVM =====")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        csp = CSP(
            n_components=n_csp_components,
            log=True,
            norm_trace=False
        )

        X_train_features = csp.fit_transform(X_train, y_train)
        X_test_features = csp.transform(X_test)

        clf = SVC(kernel="linear")

        start = time.time()
        clf.fit(X_train_features, y_train)
        end = time.time()

        train_time = end - start
        y_pred = clf.predict(X_test_features)
        acc = accuracy_score(y_test, y_pred)

        accuracies.append(acc)
        times.append(train_time)

        print(f"CSP Fold {fold} accuracy: {acc:.4f}")
        print(f"CSP Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


def run_fbcsp_svm_cv(
    X,
    y,
    groups,
    config,
    n_splits=10,
    n_csp_components=2,
    fs=250,
    k_features=8,
):
    """
    Run FBCSP + MIBIF-style feature selection + SVM.

    Pipeline:
    filter bank -> CSP per band -> concatenate features
    -> MI feature selection -> SVM
    """
    # MNE CSP expects (samples, channels, time)
    X = np.transpose(X, (0, 2, 1))

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)
    bands = get_filter_bands(config)

    accuracies = []
    times = []

    print("\n===== Running FBCSP + SVM =====")
    print("Filter bands:", bands)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_feature_list = []
        test_feature_list = []

        for lowcut, highcut in bands:
            X_train_band = cheby2_bandpass_filter_epochs(X_train, lowcut, highcut, fs=fs)
            X_test_band = cheby2_bandpass_filter_epochs(X_test, lowcut, highcut, fs=fs)

            csp = CSP(
                n_components=n_csp_components,
                log=True,
                norm_trace=False
            )

            X_train_band_features = csp.fit_transform(X_train_band, y_train)
            X_test_band_features = csp.transform(X_test_band)

            train_feature_list.append(X_train_band_features)
            test_feature_list.append(X_test_band_features)

        # Concatenate features from all bands
        X_train_features = np.concatenate(train_feature_list, axis=1)
        X_test_features = np.concatenate(test_feature_list, axis=1)

        # MIBIF-style feature selection on training fold only
        k_use = min(k_features, X_train_features.shape[1])
        X_train_sel, X_test_sel, top_idx = select_top_mibif_features(
            X_train_features, y_train, X_test_features, k_use
        )

        clf = SVC(kernel="linear")

        start = time.time()
        clf.fit(X_train_sel, y_train)
        end = time.time()

        train_time = end - start
        y_pred = clf.predict(X_test_sel)
        acc = accuracy_score(y_test, y_pred)

        accuracies.append(acc)
        times.append(train_time)

        print(f"FBCSP Fold {fold} accuracy: {acc:.4f}")
        print(f"FBCSP Fold {fold} training time: {train_time:.4f} s")
        print(f"FBCSP Fold {fold} selected feature indices: {top_idx}\n")

    return accuracies, times


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

    # CSP + SVM
    csp_accuracies, csp_times = run_csp_svm_cv(
        X,
        y,
        groups,
        config=config,
        n_splits=10,
        n_csp_components=2
    )

    # FBCSP + SVM
    fbcsp_accuracies, fbcsp_times = run_fbcsp_svm_cv(
        X,
        y,
        groups,
        config=config,
        n_splits=10,
        n_csp_components=2,
        fs=250,
        k_features=8
    )

    print("\n===== Final Comparison =====")
    print("CSP + SVM")
    print(f"Mean accuracy: {np.mean(csp_accuracies):.4f}")
    print(f"Std accuracy : {np.std(csp_accuracies):.4f}")
    print(f"Avg training time: {np.mean(csp_times):.4f} s")

    print("\nFBCSP + SVM")
    print(f"Mean accuracy: {np.mean(fbcsp_accuracies):.4f}")
    print(f"Std accuracy : {np.std(fbcsp_accuracies):.4f}")
    print(f"Avg training time: {np.mean(fbcsp_times):.4f} s")