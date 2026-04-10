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
    preprocess_subject_windows,
    preprocess_multiple_subjects,
)
from cross_validation import make_cv_splits

from fbcsp_svm import run_csp_svm_cv, run_fbcsp_svm_cv
from EEGNet import run_eegnet_cv
from DeepConvNet import run_deepconvnet_cv
from ShallowConvNet import run_shallowconvnet_cv
from EEGTCNet import run_eegtcnet_cv
from ATCNet import run_atcnet_cv


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
ALL_MODELS = ["csp", "fbcsp", "eegnet", "deepconv", "shallowconv", "eegtcnet", "atcnet"]

MODEL_DISPLAY = {
    "csp":          "CSP + SVM",
    "fbcsp":        "FBCSP + SVM",
    "eegnet":       "EEGNet",
    "deepconv":     "DeepConvNet",
    "shallowconv":  "ShallowConvNet",
    "eegtcnet":     "EEG-TCNet",
    "atcnet":       "ATCNet",
}


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
        "--n-splits", type=int, default=10,
        help="Number of CV folds. Default: 10."
    )
    args = parser.parse_args()

    config = parse_config_str(args.config)

    files = get_training_files("data")
    print(f"Found {len(files)} subject files.")
    print(f"Config: {config}")
    print(f"Models: {args.models}")
    print(f"CV folds: {args.n_splits}")

    # Load data
    if args.all_subjects:
        print("\nLoading all subjects...")
        X, y, groups = preprocess_multiple_subjects(files, config)
        data_label = "all subjects"
    else:
        idx = args.subject - 1
        if idx < 0 or idx >= len(files):
            raise ValueError(f"Subject {args.subject} out of range (1-{len(files)}).")
        print(f"\nLoading subject {args.subject} ({files[idx].name})...")
        X, y, groups = preprocess_subject_windows(files[idx], config)
        data_label = f"subject {args.subject}"

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"groups is None: {groups is None}")

    # Run models
    results = {}
    total_start = time.time()

    for name in args.models:
        display = MODEL_DISPLAY[name]
        print(f"\n{'=' * 50}")
        print(f"  Running: {display}")
        print(f"{'=' * 50}")

        accs, times = run_model(name, X, y, groups, config, args.n_splits)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)

        results[display] = (mean_acc, std_acc, mean_time)

    total_time = time.time() - total_start

    # Summary
    print(f"\nData: {data_label}")
    print_results_table(results)
    print(f"\nTotal experiment time: {total_time:.1f} s")


if __name__ == "__main__":
    main()
