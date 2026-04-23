"""
run_all.py
----------
Unified experiment runner for the MI classification comparison study.

Runs all models on a chosen subject (or all subjects) and prints a
final comparison table.

Usage:
    python run_all.py                        # single subject (B01T.mat)
    python run_all.py --subject 3            # subject 3 (B03T.mat)
    python run_all.py --all-subjects         # all 9 subjects averaged
    python run_all.py --models eegnet fbcsp  # run selected models only
    python run_all.py --config A1B2C1D2      # specify preprocessing config

Available model names:
    csp       - CSP + SVM
    fbcsp     - FBCSP + SVM
    eegnet    - EEGNet
    deepconv  - DeepConvNet
    shallowconv - ShallowConvNet
    eegtcnet  - EEG-TCNet
    atcnet    - ATCNet
"""

import argparse
import time
import numpy as np

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    get_evaluation_files,
    preprocess_subject_windows,
    preprocess_multiple_subjects,
)
from preprocessing_2a import (
    get_training_files_2a,
    get_evaluation_files_2a,
    preprocess_subject_windows_2a,
    preprocess_multiple_subjects_2a,
)
from fbcsp_svm import run_csp_svm_cv, run_fbcsp_svm_cv, run_csp_svm_holdout, run_fbcsp_svm_holdout
from EEGNet import run_eegnet_cv, run_eegnet_holdout
from DeepConvNet import run_deepconvnet_cv, run_deepconvnet_holdout
from ShallowConvNet import run_shallowconvnet_cv, run_shallowconvnet_holdout
from EEGTCNet import run_eegtcnet_cv, run_eegtcnet_holdout
from ATCNet import run_atcnet_cv, run_atcnet_holdout
from MCSANet import run_mcsanet_cv, run_mcsanet_holdout


# =========================
# Config parser
# =========================
def parse_config_str(config_str: str) -> PreprocessingConfig:
    """
    Parse a config string like 'A1B2C1D2' into a PreprocessingConfig.
    """
    import re
    m = re.fullmatch(r"A([1-4])B([12])C([12])D([12])", config_str.upper())
    if not m:
        raise ValueError(
            f"Invalid config string '{config_str}'. "
            "Expected format: A<1-4>B<1-2>C<1-2>D<1-2>, e.g. A1B2C1D2"
        )
    return PreprocessingConfig(
        A=int(m.group(1)),
        B=int(m.group(2)),
        C=int(m.group(3)),
        D=int(m.group(4)),
    )


# =========================
# Model registry
# =========================
ALL_MODELS = ["csp", "fbcsp", "eegnet", "deepconv", "shallowconv", "eegtcnet", "atcnet", "mcsanet"]

MODEL_DISPLAY = {
    "csp":          "CSP + SVM",
    "fbcsp":        "FBCSP + SVM",
    "eegnet":       "EEGNet",
    "deepconv":     "DeepConvNet",
    "shallowconv":  "ShallowConvNet",
    "eegtcnet":     "EEG-TCNet",
    "atcnet":       "ATCNet",
    "mcsanet":      "MCSANet",
}


def run_model_holdout(name, X_train, y_train, X_test, y_test, config):
    """
    Dispatch to holdout (T-E) evaluation for Dataset 2b.
    Returns (accuracy: float, time: float).
    """
    if name == "csp":
        return run_csp_svm_holdout(X_train, y_train, X_test, y_test)
    elif name == "fbcsp":
        return run_fbcsp_svm_holdout(X_train, y_train, X_test, y_test, config=config)
    elif name == "eegnet":
        return run_eegnet_holdout(X_train, y_train, X_test, y_test,
                                  epochs=50, batch_size=16, learning_rate=5e-4)
    elif name == "deepconv":
        return run_deepconvnet_holdout(X_train, y_train, X_test, y_test,
                                       epochs=50, batch_size=16, learning_rate=5e-4)
    elif name == "shallowconv":
        return run_shallowconvnet_holdout(X_train, y_train, X_test, y_test,
                                          epochs=50, batch_size=16, learning_rate=1e-3)
    elif name == "eegtcnet":
        return run_eegtcnet_holdout(X_train, y_train, X_test, y_test,
                                    epochs=80, batch_size=16, learning_rate=5e-4)
    elif name == "atcnet":
        return run_atcnet_holdout(X_train, y_train, X_test, y_test,
                                  epochs=100, batch_size=16, learning_rate=1e-3)
    elif name == "mcsanet":
        return run_mcsanet_holdout(X_train, y_train, X_test, y_test,
                                   epochs=300, batch_size=16, learning_rate=1e-3)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_model(name, X, y, groups, config, n_splits):
    """
    Dispatch to the correct run_*_cv function and return (accuracies, times).
    """
    if name == "csp":
        return run_csp_svm_cv(X, y, groups, config, n_splits=n_splits)

    elif name == "fbcsp":
        return run_fbcsp_svm_cv(X, y, groups, config, n_splits=n_splits)

    elif name == "eegnet":
        return run_eegnet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=50, batch_size=16, learning_rate=5e-4
        )

    elif name == "deepconv":
        return run_deepconvnet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=50, batch_size=16, learning_rate=5e-4
        )

    elif name == "shallowconv":
        return run_shallowconvnet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=50, batch_size=16, learning_rate=1e-3
        )

    elif name == "eegtcnet":
        return run_eegtcnet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=80, batch_size=16, learning_rate=5e-4
        )

    elif name == "atcnet":
        return run_atcnet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=100, batch_size=16, learning_rate=1e-3
        )

    elif name == "mcsanet":
        return run_mcsanet_cv(
            X, y, groups, config, n_splits=n_splits,
            epochs=300, batch_size=16, learning_rate=1e-3
        )

    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# Result table
# =========================
def print_results_table(results: dict):
    """
    Print a formatted comparison table.

    results : dict mapping display_name -> (mean_acc, std_acc, mean_time)
    """
    print("\n" + "=" * 62)
    print("  FINAL COMPARISON TABLE")
    print("=" * 62)
    print(f"  {'Model':<20} {'Acc (mean)':<14} {'Acc (std)':<14} {'Avg Time (s)'}")
    print("-" * 62)

    for display_name, (mean_acc, std_acc, mean_time) in results.items():
        print(
            f"  {display_name:<20} "
            f"{mean_acc * 100:>8.2f} %     "
            f"{std_acc * 100:>7.2f} %     "
            f"{mean_time:>8.3f}"
        )

    print("=" * 62)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Run all MI classification models and compare results."
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="Subject number to run (1-9). Default: 1. Ignored if --all-subjects is set."
    )
    parser.add_argument(
        "--all-subjects", action="store_true",
        help="Run on all subjects and average results."
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Which models to run. Default: all."
    )
    parser.add_argument(
        "--config", type=str, default="A1B2C1D2",
        help="Preprocessing config string, e.g. A1B2C1D2. Default: A1B2C1D2."
    )
    parser.add_argument(
        "--dataset", type=str, default="2b", choices=["2b", "2a"],
        help="Which dataset to use: 2b (default) or 2a."
    )
    parser.add_argument(
        "--n-splits", type=int, default=10,
        help="Number of CV folds. Default: 10."
    )
    args = parser.parse_args()

    config = parse_config_str(args.config)

    print(f"Dataset: {args.dataset}")
    print(f"Config: {config}")
    print(f"Models: {args.models}")

    total_start = time.time()
    results = {}

    # ── Dataset 2b: official T-E holdout protocol ──────────────────────────
    if args.dataset == "2b":
        t_files = get_training_files("data/2b")
        e_files = get_evaluation_files("data/2b")

        if args.all_subjects:
            print(f"\nRunning T-E holdout on all {len(t_files)} subjects...")
            subj_accs  = {n: [] for n in args.models}
            subj_times = {n: [] for n in args.models}

            for i, (tf, ef) in enumerate(zip(t_files, e_files), start=1):
                print(f"\n--- Subject {i} ---")
                X_train, y_train, _ = preprocess_subject_windows(tf, config)
                X_test,  y_test,  _ = preprocess_subject_windows(ef, config)
                print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}")
                    print(f"  Running: {display}")
                    print(f"{'=' * 50}")
                    acc, t = run_model_holdout(name, X_train, y_train, X_test, y_test, config)
                    subj_accs[name].append(acc)
                    subj_times[name].append(t)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_times[name]),
                )
            data_label = "all subjects (T-E)"

        else:
            idx = args.subject - 1
            if idx < 0 or idx >= len(t_files):
                raise ValueError(f"Subject {args.subject} out of range (1-{len(t_files)}).")
            print(f"\nLoading subject {args.subject} ...")
            X_train, y_train, _ = preprocess_subject_windows(t_files[idx], config)
            X_test,  y_test,  _ = preprocess_subject_windows(e_files[idx], config)
            print(f"Train: {X_train.shape}  Test: {X_test.shape}")

            for name in args.models:
                display = MODEL_DISPLAY[name]
                print(f"\n{'=' * 50}")
                print(f"  Running: {display}")
                print(f"{'=' * 50}")
                acc, t = run_model_holdout(name, X_train, y_train, X_test, y_test, config)
                results[display] = (acc, 0.0, t)

            data_label = f"subject {args.subject} (T-E)"

    # ── Dataset 2a: official T-E holdout protocol ──────────────────────────
    else:
        t_files = get_training_files_2a("data/2a")
        e_files = get_evaluation_files_2a("data/2a")

        if args.all_subjects:
            print(f"\nRunning T-E holdout on all {len(t_files)} subjects...")
            subj_accs  = {n: [] for n in args.models}
            subj_times = {n: [] for n in args.models}

            for i, (tf, ef) in enumerate(zip(t_files, e_files), start=1):
                print(f"\n--- Subject {i} ---")
                X_train, y_train, _ = preprocess_subject_windows_2a(tf, config)
                X_test,  y_test,  _ = preprocess_subject_windows_2a(ef, config)
                print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}")
                    print(f"  Running: {display}")
                    print(f"{'=' * 50}")
                    acc, t = run_model_holdout(name, X_train, y_train, X_test, y_test, config)
                    subj_accs[name].append(acc)
                    subj_times[name].append(t)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_times[name]),
                )
            data_label = "all subjects (T-E)"

        else:
            idx = args.subject - 1
            if idx < 0 or idx >= len(t_files):
                raise ValueError(f"Subject {args.subject} out of range (1-{len(t_files)}).")
            print(f"\nLoading subject {args.subject} ...")
            X_train, y_train, _ = preprocess_subject_windows_2a(t_files[idx], config)
            X_test,  y_test,  _ = preprocess_subject_windows_2a(e_files[idx], config)
            print(f"Train: {X_train.shape}  Test: {X_test.shape}")

            for name in args.models:
                display = MODEL_DISPLAY[name]
                print(f"\n{'=' * 50}")
                print(f"  Running: {display}")
                print(f"{'=' * 50}")
                acc, t = run_model_holdout(name, X_train, y_train, X_test, y_test, config)
                results[display] = (acc, 0.0, t)

            data_label = f"subject {args.subject} (T-E)"

    total_time = time.time() - total_start

    # Summary
    print(f"\nData: {data_label}")
    print_results_table(results)
    print(f"\nTotal experiment time: {total_time:.1f} s")


if __name__ == "__main__":
    main()
