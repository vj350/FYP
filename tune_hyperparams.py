"""
tune_hyperparams.py
-------------------
Hyperparameter tuning for all 7 MI classification models.

Strategy:
- ML models  (CSP+SVM, FBCSP+SVM) : manual grid search with 5-fold CV
- DL models  (EEGNet, DeepConvNet, ShallowConvNet, EEGTCNet, ATCNet)
             : grid search over a single 80/20 train/val split
             (full CV per combination would take too long)

Usage:
    python tune_hyperparams.py                   # tune all models, subject 1
    python tune_hyperparams.py --subject 2
    python tune_hyperparams.py --models eegnet deepconv
    python tune_hyperparams.py --config A1B2C1D2
"""

import argparse
import itertools
import time
import re

import numpy as np
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from mne.decoding import CSP

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)
from fbcsp_svm import cheby2_bandpass_filter_epochs, get_filter_bands, select_top_mibif_features
from EEGModels import EEGNet, DeepConvNet, ShallowConvNet
from EEGTCNet import EEGTCNet
from ATCNet import ATCNet


# =========================
# Helpers
# =========================
def parse_config_str(config_str: str) -> PreprocessingConfig:
    m = re.fullmatch(r"A([1-4])B([12])C([12])D([12])", config_str.upper())
    if not m:
        raise ValueError(f"Invalid config string '{config_str}'. Expected e.g. A1B2C1D2")
    return PreprocessingConfig(A=int(m.group(1)), B=int(m.group(2)),
                               C=int(m.group(3)), D=int(m.group(4)))


def grid(param_dict: dict) -> list:
    """Return list of dicts for all combinations in param_dict."""
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def print_best(model_name: str, best_params: dict, best_score: float):
    print(f"\n{'=' * 55}")
    print(f"  BEST HYPERPARAMETERS — {model_name}")
    print(f"  Best val accuracy: {best_score * 100:.2f}%")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"{'=' * 55}\n")


# =========================
# ML tuning (5-fold CV)
# =========================
def tune_csp_svm(X_raw, y, groups, config, n_splits=5):
    """
    Grid search for CSP + SVM.
    Params: n_csp_components, SVM C
    """
    print("\n===== Tuning CSP + SVM =====")

    X = np.transpose(X_raw, (0, 2, 1))  # (N, T, C) -> (N, C, T)

    param_grid = grid({
        "n_csp_components": [2, 4, 6],
        "C": [0.1, 1.0, 10.0],
    })

    best_score = -1
    best_params = {}

    for params in param_grid:
        fold_accs = []

        if config.A == 1:
            cv = KFold(n_splits=n_splits, shuffle=False)
            splits = list(cv.split(X, y))
        else:
            cv = GroupKFold(n_splits=n_splits)
            splits = list(cv.split(X, y, groups))

        for train_idx, test_idx in splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            csp = CSP(n_components=params["n_csp_components"], log=True, norm_trace=False)
            X_tr_f = csp.fit_transform(X_tr, y_tr)
            X_te_f = csp.transform(X_te)

            clf = SVC(kernel="linear", C=params["C"])
            clf.fit(X_tr_f, y_tr)
            fold_accs.append(accuracy_score(y_te, clf.predict(X_te_f)))

        mean_acc = np.mean(fold_accs)
        print(f"  {params}  ->  {mean_acc * 100:.2f}%")

        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params

    print_best("CSP + SVM", best_params, best_score)
    return best_params, best_score


def tune_fbcsp_svm(X_raw, y, groups, config, n_splits=5):
    """
    Grid search for FBCSP + SVM.
    Params: n_csp_components, k_features, SVM C
    """
    print("\n===== Tuning FBCSP + SVM =====")

    X = np.transpose(X_raw, (0, 2, 1))
    bands = get_filter_bands(config)

    param_grid = grid({
        "n_csp_components": [2, 4],
        "k_features": [4, 8, 12],
        "C": [0.1, 1.0, 10.0],
    })

    best_score = -1
    best_params = {}

    for params in param_grid:
        fold_accs = []

        if config.A == 1:
            cv = KFold(n_splits=n_splits, shuffle=False)
            splits = list(cv.split(X, y))
        else:
            cv = GroupKFold(n_splits=n_splits)
            splits = list(cv.split(X, y, groups))

        for train_idx, test_idx in splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            train_feats, test_feats = [], []
            for low, high in bands:
                X_tr_b = cheby2_bandpass_filter_epochs(X_tr, low, high)
                X_te_b = cheby2_bandpass_filter_epochs(X_te, low, high)

                csp = CSP(n_components=params["n_csp_components"], log=True, norm_trace=False)
                train_feats.append(csp.fit_transform(X_tr_b, y_tr))
                test_feats.append(csp.transform(X_te_b))

            X_tr_f = np.concatenate(train_feats, axis=1)
            X_te_f = np.concatenate(test_feats, axis=1)

            k_use = min(params["k_features"], X_tr_f.shape[1])
            X_tr_sel, X_te_sel, _ = select_top_mibif_features(X_tr_f, y_tr, X_te_f, k_use)

            clf = SVC(kernel="linear", C=params["C"])
            clf.fit(X_tr_sel, y_tr)
            fold_accs.append(accuracy_score(y_te, clf.predict(X_te_sel)))

        mean_acc = np.mean(fold_accs)
        print(f"  {params}  ->  {mean_acc * 100:.2f}%")

        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params

    print_best("FBCSP + SVM", best_params, best_score)
    return best_params, best_score


# =========================
# DL helpers
# =========================
def dl_train_eval(model, X_tr, y_tr, X_val, y_val, epochs, batch_size, lr, patience=10):
    """Train a DL model on train split, return val accuracy."""
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=patience,
                               restore_best_weights=True, verbose=0)
    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0,
    )
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    return accuracy_score(y_val, y_pred)


def prepare_nct1(X, y):
    """(N,T,C) -> (N,C,T,1), labels {1,2}->{0,1}"""
    X = np.transpose(X, (0, 2, 1))[..., np.newaxis]
    y = y.astype(int) - 1
    return X.astype(np.float32), y.astype(np.int64)


def prepare_n1ct(X, y):
    """(N,T,C) -> (N,1,C,T), labels {1,2}->{0,1}"""
    X = np.transpose(X, (0, 2, 1))[:, np.newaxis, :, :]
    y = y.astype(int) - 1
    return X.astype(np.float32), y.astype(np.int64)


def dl_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Train/val split that respects groups (no trial leakage).
    If groups is None, use plain random split.
    """
    if groups is None:
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)

    # Group-aware split: hold out last ~20% of unique trial IDs
    unique_groups = np.unique(groups)
    n_val = max(1, int(len(unique_groups) * test_size))
    val_groups = unique_groups[-n_val:]

    val_mask = np.isin(groups, val_groups)
    train_mask = ~val_mask

    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


# =========================
# DL tuning
# =========================
def tune_eegnet(X_raw, y, groups, epochs=50, batch_size=16):
    print("\n===== Tuning EEGNet =====")

    X, y = prepare_nct1(X_raw, y)
    X_tr, X_val, y_tr, y_val = dl_split(X, y, groups)

    n_classes = len(np.unique(y))
    Chans, Samples = X.shape[1], X.shape[2]

    param_grid = grid({
        "dropoutRate": [0.25, 0.5],
        "kernLength":  [64, 128],
        "F1":          [8, 16],
        "learning_rate": [1e-3, 5e-4],
    })

    best_score, best_params = -1, {}

    for params in param_grid:
        model = EEGNet(
            nb_classes=n_classes, Chans=Chans, Samples=Samples,
            dropoutRate=params["dropoutRate"],
            kernLength=params["kernLength"],
            F1=params["F1"], D=2, F2=params["F1"] * 2,
            norm_rate=0.25, dropoutType="Dropout",
        )
        acc = dl_train_eval(model, X_tr, y_tr, X_val, y_val,
                            epochs, batch_size, params["learning_rate"])
        print(f"  {params}  ->  {acc * 100:.2f}%")
        if acc > best_score:
            best_score, best_params = acc, params

    print_best("EEGNet", best_params, best_score)
    return best_params, best_score


def tune_deepconvnet(X_raw, y, groups, epochs=50, batch_size=16):
    print("\n===== Tuning DeepConvNet =====")

    X, y = prepare_nct1(X_raw, y)
    X_tr, X_val, y_tr, y_val = dl_split(X, y, groups)

    n_classes = len(np.unique(y))
    Chans, Samples = X.shape[1], X.shape[2]

    param_grid = grid({
        "dropoutRate":   [0.25, 0.5],
        "learning_rate": [1e-3, 5e-4],
    })

    best_score, best_params = -1, {}

    for params in param_grid:
        model = DeepConvNet(
            nb_classes=n_classes, Chans=Chans, Samples=Samples,
            dropoutRate=params["dropoutRate"],
        )
        acc = dl_train_eval(model, X_tr, y_tr, X_val, y_val,
                            epochs, batch_size, params["learning_rate"])
        print(f"  {params}  ->  {acc * 100:.2f}%")
        if acc > best_score:
            best_score, best_params = acc, params

    print_best("DeepConvNet", best_params, best_score)
    return best_params, best_score


def tune_shallowconvnet(X_raw, y, groups, epochs=50, batch_size=16):
    print("\n===== Tuning ShallowConvNet =====")

    X, y = prepare_nct1(X_raw, y)
    X_tr, X_val, y_tr, y_val = dl_split(X, y, groups)

    n_classes = len(np.unique(y))
    Chans, Samples = X.shape[1], X.shape[2]

    param_grid = grid({
        "dropoutRate":   [0.25, 0.5],
        "learning_rate": [1e-3, 5e-4],
    })

    best_score, best_params = -1, {}

    for params in param_grid:
        model = ShallowConvNet(
            nb_classes=n_classes, Chans=Chans, Samples=Samples,
            dropoutRate=params["dropoutRate"],
        )
        acc = dl_train_eval(model, X_tr, y_tr, X_val, y_val,
                            epochs, batch_size, params["learning_rate"])
        print(f"  {params}  ->  {acc * 100:.2f}%")
        if acc > best_score:
            best_score, best_params = acc, params

    print_best("ShallowConvNet", best_params, best_score)
    return best_params, best_score


def tune_eegtcnet(X_raw, y, groups, epochs=80, batch_size=16):
    print("\n===== Tuning EEG-TCNet =====")

    X, y = prepare_n1ct(X_raw, y)
    X_tr, X_val, y_tr, y_val = dl_split(X, y, groups)

    n_classes = len(np.unique(y))
    Chans, Samples = X.shape[2], X.shape[3]

    param_grid = grid({
        "filt":          [12, 24],
        "dropout":       [0.2, 0.3],
        "learning_rate": [1e-3, 5e-4],
    })

    best_score, best_params = -1, {}

    for params in param_grid:
        model = EEGTCNet(
            n_classes=n_classes, Chans=Chans, Samples=Samples,
            layers=2, kernel_s=4,
            filt=params["filt"],
            dropout=params["dropout"],
            activation="elu", F1=8, D=2, kernLength=32, dropout_eeg=0.2,
        )
        acc = dl_train_eval(model, X_tr, y_tr, X_val, y_val,
                            epochs, batch_size, params["learning_rate"],
                            patience=12)
        print(f"  {params}  ->  {acc * 100:.2f}%")
        if acc > best_score:
            best_score, best_params = acc, params

    print_best("EEG-TCNet", best_params, best_score)
    return best_params, best_score


def tune_atcnet(X_raw, y, groups, epochs=100, batch_size=16):
    print("\n===== Tuning ATCNet =====")

    X, y = prepare_n1ct(X_raw, y)
    X_tr, X_val, y_tr, y_val = dl_split(X, y, groups)

    n_classes = len(np.unique(y))
    Chans, Samples = X.shape[2], X.shape[3]

    param_grid = grid({
        "eegn_F1":       [8, 16],
        "tcn_filters":   [16, 32],
        "learning_rate": [1e-3, 5e-4],
    })

    best_score, best_params = -1, {}

    for params in param_grid:
        model = ATCNet(
            n_classes=n_classes, in_chans=Chans, in_samples=Samples,
            n_windows=5,
            eegn_F1=params["eegn_F1"], eegn_D=2,
            eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4,
            tcn_filters=params["tcn_filters"],
            tcn_dropout=0.3, tcn_activation="elu", fuse="average",
        )
        acc = dl_train_eval(model, X_tr, y_tr, X_val, y_val,
                            epochs, batch_size, params["learning_rate"],
                            patience=15)
        print(f"  {params}  ->  {acc * 100:.2f}%")
        if acc > best_score:
            best_score, best_params = acc, params

    print_best("ATCNet", best_params, best_score)
    return best_params, best_score


# =========================
# Summary table
# =========================
def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)
    for model_name, (best_params, best_score) in results.items():
        print(f"\n  {model_name}  (val acc: {best_score * 100:.2f}%)")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
    print("\n" + "=" * 60)
    print("  Copy best params into run_all.py for final experiments.")
    print("=" * 60)


# =========================
# Main
# =========================
ALL_MODELS = ["csp", "fbcsp", "eegnet", "deepconv", "shallowconv", "eegtcnet", "atcnet"]

TUNE_FN = {
    "csp":         lambda X, y, g, cfg: tune_csp_svm(X, y, g, cfg),
    "fbcsp":       lambda X, y, g, cfg: tune_fbcsp_svm(X, y, g, cfg),
    "eegnet":      lambda X, y, g, cfg: tune_eegnet(X, y, g),
    "deepconv":    lambda X, y, g, cfg: tune_deepconvnet(X, y, g),
    "shallowconv": lambda X, y, g, cfg: tune_shallowconvnet(X, y, g),
    "eegtcnet":    lambda X, y, g, cfg: tune_eegtcnet(X, y, g),
    "atcnet":      lambda X, y, g, cfg: tune_atcnet(X, y, g),
}

MODEL_DISPLAY = {
    "csp": "CSP + SVM", "fbcsp": "FBCSP + SVM",
    "eegnet": "EEGNet", "deepconv": "DeepConvNet",
    "shallowconv": "ShallowConvNet", "eegtcnet": "EEG-TCNet", "atcnet": "ATCNet",
}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for all MI models.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--config", type=str, default="A1B2C1D2")
    args = parser.parse_args()

    config = parse_config_str(args.config)
    files = get_training_files("data")

    idx = args.subject - 1
    print(f"Tuning on subject {args.subject} ({files[idx].name})")
    print(f"Config: {config}")
    print(f"Models: {args.models}\n")

    X, y, groups = preprocess_subject_windows(files[idx], config)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    results = {}
    total_start = time.time()

    for name in args.models:
        display = MODEL_DISPLAY[name]
        best_params, best_score = TUNE_FN[name](X, y, groups, config)
        results[display] = (best_params, best_score)

    print(f"\nTotal tuning time: {time.time() - total_start:.1f} s")
    print_summary(results)


if __name__ == "__main__":
    main()
