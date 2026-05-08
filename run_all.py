"""
run_all.py
----------
Unified experiment runner for the MI classification comparison study.

Usage:
    python run_all.py --protocol te --all-subjects --dataset 2a   # T-E all subjects
    python run_all.py --protocol loso --dataset 2a                # LOSO all subjects
    python run_all.py --protocol te --subject 1 --dataset 2b      # T-E single subject
    python run_all.py --models eegnet fbcsp --dataset 2a          # selected models
    python run_all.py --config A1B2C1D2 --dataset 2b              # custom config

Protocols:
    cv   - 10-fold stratified cross-validation on T files
    te   - Train on T file(s), evaluate on E file(s) — standard competition split
    loso - Leave-One-Subject-Out cross-validation (subject-independent evaluation)

Available model names:
    csp         - CSP + SVM
    fbcsp       - FBCSP + SVM
    eegnet      - EEGNet
    deepconv    - DeepConvNet
    shallowconv - ShallowConvNet
    atcnet      - ATCNet
    mcsanet     - MCSANet
"""

import argparse
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    get_evaluation_files,
    preprocess_subject_windows,
)
from preprocessing_2a import (
    get_training_files_2a,
    get_evaluation_files_2a,
    preprocess_subject_windows_2a,
)
from fbcsp_svm import run_csp_svm_holdout, run_fbcsp_svm_holdout
from EEGNet import run_eegnet_holdout
from DeepConvNet import run_deepconvnet_holdout
from ShallowConvNet import run_shallowconvnet_holdout
from ATCNet import run_atcnet_holdout
from MCSANet import run_mcsanet_holdout


# =========================
# Config parser
# =========================
def parse_config_str(config_str: str) -> PreprocessingConfig:
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
ALL_MODELS = ["csp", "fbcsp", "eegnet", "deepconv", "shallowconv",
              "atcnet", "mcsanet"]

MODEL_DISPLAY = {
    "csp":          "CSP + SVM",
    "fbcsp":        "FBCSP + SVM",
    "eegnet":       "EEGNet",
    "deepconv":     "DeepConvNet",
    "shallowconv":  "ShallowConvNet",
    "atcnet":       "ATCNet",
    "mcsanet":      "MCSANet",
}

# Models that use no bandpass filter (raw signal passed in)
# MCSANet paper explicitly states raw signal — no bandpass filter
# All other DL models use the broad bandpass from config (C/D factors)
MODELS_NO_FILTER = {"eegnet", "deepconv", "shallowconv", "atcnet", "mcsanet"}


def run_model_holdout(name, X_train, y_train, X_test, y_test, config, augment=True):
    """
    Dispatch to the correct holdout evaluation function.
    Returns (accuracy: float, kappa: float, time: float).
    Used both for T-E holdout and for individual CV folds.
    """
    if name == "csp":
        return run_csp_svm_holdout(X_train, y_train, X_test, y_test, augment=augment)
    elif name == "fbcsp":
        return run_fbcsp_svm_holdout(X_train, y_train, X_test, y_test,
                                     config=config, n_csp_components=4,
                                     k_features=16, augment=augment)
    elif name == "eegnet":
        return run_eegnet_holdout(X_train, y_train, X_test, y_test,
                                  epochs=500, batch_size=16, learning_rate=1e-3,
                                  augment=augment)
    elif name == "deepconv":
        return run_deepconvnet_holdout(X_train, y_train, X_test, y_test,
                                       epochs=500, batch_size=16, learning_rate=1e-3,
                                       augment=augment)
    elif name == "shallowconv":
        return run_shallowconvnet_holdout(X_train, y_train, X_test, y_test,
                                          epochs=500, batch_size=16, learning_rate=1e-3,
                                          augment=augment)
    elif name == "atcnet":
        return run_atcnet_holdout(X_train, y_train, X_test, y_test,
                                  epochs=500, batch_size=64, learning_rate=1e-3)
    elif name == "mcsanet":
        return run_mcsanet_holdout(X_train, y_train, X_test, y_test,
                                   epochs=500, batch_size=16, learning_rate=1e-3,
                                   augment=augment)
    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# 10-fold CV for one subject
# =========================
def run_subject_cv(name, X, y, config, n_splits=10, augment=True):
    """
    Run n_splits-fold stratified CV for one subject.
    Returns (mean_accuracy, mean_kappa, mean_time_per_fold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs   = []
    fold_kappas = []
    fold_times  = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        acc, kappa, t = run_model_holdout(
            name,
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            config,
            augment=augment,
        )
        fold_accs.append(acc)
        fold_kappas.append(kappa)
        fold_times.append(t)
    return np.mean(fold_accs), np.mean(fold_kappas), np.mean(fold_times)


# =========================
# Result table
# =========================
def print_results_table(results: dict, per_subject: dict = None):
    """
    Print a formatted comparison table.
    results     : dict mapping display_name -> (mean_acc, std_acc, mean_kappa, std_kappa, mean_time)
    per_subject : optional dict mapping display_name -> [(acc, kappa), ...]
    """
    models = list(results.keys())

    # Per-subject breakdown
    if per_subject:
        n_subjects = len(next(iter(per_subject.values())))
        header = f"  {'Subject':<10}" + "".join(f"  {m:<28}" for m in models)
        print("\n" + "=" * (10 + 30 * len(models)))
        print("  PER-SUBJECT RESULTS")
        print("=" * (10 + 30 * len(models)))
        print(header)
        print("-" * (10 + 30 * len(models)))
        for i in range(n_subjects):
            row = f"  {f'sub{i+1}':<10}"
            for m in models:
                acc, kap = per_subject[m][i]
                row += f"  {acc*100:>6.2f}% (κ={kap:.2f})        "
            print(row)
        print("-" * (10 + 30 * len(models)))

    # Summary table
    print("\n" + "=" * 76)
    print("  FINAL COMPARISON TABLE")
    print("=" * 76)
    print(f"  {'Model':<20} {'Acc (mean)':<14} {'Acc (std)':<12} {'Kappa (mean)':<14} {'Avg Time (s)'}")
    print("-" * 76)
    for display_name, vals in results.items():
        mean_acc, std_acc, mean_kappa, std_kappa, mean_time = vals
        print(
            f"  {display_name:<20} "
            f"{mean_acc * 100:>8.2f} %   "
            f"{std_acc * 100:>6.2f} %   "
            f"{mean_kappa:>8.2f}        "
            f"{mean_time:>8.3f}"
        )
    print("=" * 76)


def save_partial_result(dataset: str, protocol: str, subject_idx: int,
                        model_name: str, acc: float, kappa: float):
    """
    Append a single model result for one subject immediately after it completes.
    Acts as a checkpoint in case the run is interrupted.
    """
    filename = f"results_{dataset}_{protocol}_checkpoint.txt"
    line = f"Subject {subject_idx+1:2d} | {model_name:<20} | acc={acc*100:.2f}%  kappa={kappa:.2f}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line)


def load_completed_results(dataset: str, protocol: str):
    """
    Read checkpoint file and return:
      - completed : set of (subject_idx, model_display_name) already done
      - saved     : dict mapping (subject_idx, model_display_name) -> (acc, kappa)
    Used to skip already-done work and restore results when resuming an interrupted run.
    """
    filename = f"results_{dataset}_{protocol}_checkpoint.txt"
    completed = set()
    saved = {}
    if not os.path.exists(filename):
        return completed, saved
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Format: "Subject  1 | ModelName            | acc=XX.XX%  kappa=X.XX"
                parts = line.split("|")
                subj_idx = int(parts[0].replace("Subject", "").strip()) - 1
                model_name = parts[1].strip()
                acc_str = parts[2].split("acc=")[1].split("%")[0]
                kappa_str = parts[2].split("kappa=")[1]
                acc = float(acc_str) / 100.0
                kappa = float(kappa_str)
                completed.add((subj_idx, model_name))
                saved[(subj_idx, model_name)] = (acc, kappa)
            except Exception:
                continue
    return completed, saved


def save_results_to_file(results: dict, data_label: str, dataset: str,
                         protocol: str, total_time: float,
                         per_subject: dict = None):
    """
    Save results to a text file (results_2a.txt or results_2b.txt).
    """
    filename = f"results_{dataset}.txt"
    lines = []
    models = list(results.keys())
    col = 12  # column width per model

    if per_subject:
        n_subjects = len(next(iter(per_subject.values())))
        # Header
        header = f"{'Subject':<10}" + "".join(f"{m:>{col}}" for m in models)
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)
        for i in range(n_subjects):
            row = f"{'S' + str(i+1):<10}"
            for m in models:
                acc, _ = per_subject[m][i]
                row += f"{acc*100:>10.2f}%  "
            lines.append(row)
        lines.append(sep)
        lines.append("")

    # Overall summary
    lines.append("Overall (mean +/- std):")
    lines.append("")
    hdr2 = f"{'Model':<18} {'Accuracy':>22}   {'k':>6}   {'Avg Time':>10}"
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for display_name, vals in results.items():
        mean_acc, std_acc, mean_kappa, std_kappa, mean_time = vals
        lines.append(
            f"{display_name:<18} "
            f"{mean_acc*100:>8.2f}% +/- {std_acc*100:>6.2f}%"
            f"   {mean_kappa:>6.2f}"
            f"   {mean_time:>8.0f}s"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults saved to {filename}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Run MI classification models with 10-fold CV."
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="Subject number (1-9). Default: 1. Ignored if --all-subjects is set."
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
    parser.add_argument(
        "--protocol", type=str, default="te", choices=["cv", "te", "loso"],
        help="Evaluation protocol: te (T-E holdout), loso (subject-independent), cv (10-fold CV). Default: te."
    )
    args = parser.parse_args()

    config = parse_config_str(args.config)

    print(f"Dataset  : {args.dataset}")
    print(f"Protocol : {args.protocol.upper()}")
    print(f"Config   : {config}")
    print(f"Models   : {args.models}")
    if args.protocol == "cv":
        print(f"CV folds : {args.n_splits}")

    total_start = time.time()
    results     = {}
    per_subject = None

    # ── Dataset 2b ────────────────────────────────────────────────────────
    if args.dataset == "2b":
        t_files = get_training_files("data/2b")
        e_files = get_evaluation_files("data/2b")

        # ── T-E holdout ───────────────────────────────────────────────────
        if args.protocol == "te":
            if args.all_subjects:
                print(f"\nRunning T-E holdout on all {len(t_files)} subjects...")
                # Clear checkpoint file so this run starts fresh
                open(f"results_{args.dataset}_{args.protocol}_checkpoint.txt", "w", encoding="utf-8").close()
                subj_accs   = {n: [] for n in args.models}
                subj_kappas = {n: [] for n in args.models}
                subj_times  = {n: [] for n in args.models}
                per_subject = {MODEL_DISPLAY[n]: [] for n in args.models}

                for i, (tf, ef) in enumerate(zip(t_files, e_files), start=1):
                    print(f"\n--- Subject {i} ---")
                    X_tr, y_tr, _ = preprocess_subject_windows(tf, config)
                    X_te, y_te, _ = preprocess_subject_windows(ef, config)
                    needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                    if needs_raw:
                        X_tr_raw, _, _ = preprocess_subject_windows(tf, config, apply_filter=False)
                        X_te_raw, _, _ = preprocess_subject_windows(ef, config, apply_filter=False)

                    for name in args.models:
                        display = MODEL_DISPLAY[name]
                        print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                        Xtr = X_tr_raw if name in MODELS_NO_FILTER else X_tr
                        Xte = X_te_raw if name in MODELS_NO_FILTER else X_te
                        acc, kappa, t = run_model_holdout(name, Xtr, y_tr, Xte, y_te, config, augment=True)
                        print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                        subj_accs[name].append(acc)
                        subj_kappas[name].append(kappa)
                        subj_times[name].append(t)
                        per_subject[display].append((acc, kappa))
                        save_partial_result(args.dataset, args.protocol, i-1, display, acc, kappa)

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    results[display] = (
                        np.mean(subj_accs[name]),
                        np.std(subj_accs[name]),
                        np.mean(subj_kappas[name]),
                        np.std(subj_kappas[name]),
                        np.mean(subj_times[name]),
                    )
                data_label = "all subjects (T-E holdout)"

            else:
                idx = args.subject - 1
                print(f"\nLoading subject {args.subject} (T-E holdout)...")
                X_tr, y_tr, _ = preprocess_subject_windows(t_files[idx], config)
                X_te, y_te, _ = preprocess_subject_windows(e_files[idx], config)
                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_tr_raw, _, _ = preprocess_subject_windows(t_files[idx], config, apply_filter=False)
                    X_te_raw, _, _ = preprocess_subject_windows(e_files[idx], config, apply_filter=False)
                print(f"Train trials: {X_tr.shape[0]}  Test trials: {X_te.shape[0]}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                    Xtr = X_tr_raw if name in MODELS_NO_FILTER else X_tr
                    Xte = X_te_raw if name in MODELS_NO_FILTER else X_te
                    acc, kappa, t = run_model_holdout(name, Xtr, y_tr, Xte, y_te, config, augment=True)
                    print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    results[display] = (acc, 0.0, kappa, 0.0, t)
                data_label = f"subject {args.subject} (T-E holdout)"

        # ── LOSO ─────────────────────────────────────────────────────────
        elif args.protocol == "loso":
            loso_indices = list(range(len(t_files))) if args.all_subjects else [args.subject - 1]
            if args.all_subjects:
                print(f"\nRunning LOSO on all {len(t_files)} subjects (2b)...")
                completed, saved = load_completed_results(args.dataset, args.protocol)
                if completed:
                    print(f"  Resuming — {len(completed)} results already in checkpoint, skipping those.")
                else:
                    open(f"results_{args.dataset}_{args.protocol}_checkpoint.txt", "w", encoding="utf-8").close()
                    saved = {}
            else:
                print(f"\nRunning LOSO fold for subject {args.subject} (2b)...")
                completed, saved = set(), {}
            subj_accs   = {n: [] for n in args.models}
            subj_kappas = {n: [] for n in args.models}
            subj_times  = {n: [] for n in args.models}
            per_subject = {MODEL_DISPLAY[n]: [] for n in args.models}
            # Pre-populate results from checkpoint
            for (sidx, mdisplay), (acc, kappa) in saved.items():
                for n in args.models:
                    if MODEL_DISPLAY[n] == mdisplay:
                        subj_accs[n].append(acc)
                        subj_kappas[n].append(kappa)
                        subj_times[n].append(0.0)
                        per_subject[mdisplay].append((acc, kappa))

            for test_idx in loso_indices:
                print(f"\n--- LOSO: Test subject {test_idx+1} ---")
                train_files = [f for i, f in enumerate(t_files) if i != test_idx]

                X_tr_parts, y_tr_parts = [], []
                for f in train_files:
                    X_, y_, _ = preprocess_subject_windows(f, config)
                    X_tr_parts.append(X_)
                    y_tr_parts.append(y_)
                X_train = np.concatenate(X_tr_parts, axis=0)
                y_train = np.concatenate(y_tr_parts, axis=0)

                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_tr_raw_parts = []
                    for f in train_files:
                        X_raw_, _, _ = preprocess_subject_windows(f, config, apply_filter=False)
                        X_tr_raw_parts.append(X_raw_)
                    X_train_raw = np.concatenate(X_tr_raw_parts, axis=0)

                X_test, y_test, _ = preprocess_subject_windows(e_files[test_idx], config)
                if needs_raw:
                    X_test_raw, _, _ = preprocess_subject_windows(e_files[test_idx], config, apply_filter=False)

                print(f"  Train: {X_train.shape[0]} trials  Test: {X_test.shape[0]} trials")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    if (test_idx, display) in completed:
                        print(f"  Skipping {display} (already done)")
                        continue
                    print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                    Xtr = X_train_raw if name in MODELS_NO_FILTER else X_train
                    Xte = X_test_raw if name in MODELS_NO_FILTER else X_test
                    acc, kappa, t = run_model_holdout(name, Xtr, y_train, Xte, y_test, config, augment=True)
                    print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    subj_accs[name].append(acc)
                    subj_kappas[name].append(kappa)
                    subj_times[name].append(t)
                    per_subject[display].append((acc, kappa))
                    save_partial_result(args.dataset, args.protocol, test_idx, display, acc, kappa)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_kappas[name]),
                    np.std(subj_kappas[name]),
                    np.mean(subj_times[name]),
                )
            data_label = f"subject {args.subject} (LOSO fold)" if not args.all_subjects else "all subjects (LOSO)"

        # ── 10-fold CV ────────────────────────────────────────────────────
        elif args.all_subjects:
            print(f"\nRunning {args.n_splits}-fold CV on all {len(t_files)} subjects...")
            subj_accs   = {n: [] for n in args.models}
            subj_kappas = {n: [] for n in args.models}
            subj_times  = {n: [] for n in args.models}

            for i, tf in enumerate(t_files, start=1):
                print(f"\n--- Subject {i} ---")
                X, y, _ = preprocess_subject_windows(tf, config)
                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_raw, _, _ = preprocess_subject_windows(tf, config, apply_filter=False)
                print(f"  Trials: {X.shape[0]}  Shape: {X.shape}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}")
                    print(f"  Running: {display}")
                    print(f"{'=' * 50}")
                    X_use = X_raw if name in MODELS_NO_FILTER else X
                    acc, kappa, t = run_subject_cv(name, X_use, y, config,
                                                   n_splits=args.n_splits, augment=True)
                    print(f"  {display} CV mean: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    subj_accs[name].append(acc)
                    subj_kappas[name].append(kappa)
                    subj_times[name].append(t)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_kappas[name]),
                    np.std(subj_kappas[name]),
                    np.mean(subj_times[name]),
                )
            data_label = f"all subjects ({args.n_splits}-fold CV)"

        else:
            idx = args.subject - 1
            if idx < 0 or idx >= len(t_files):
                raise ValueError(f"Subject {args.subject} out of range (1-{len(t_files)}).")
            print(f"\nLoading subject {args.subject} ...")
            X, y, _ = preprocess_subject_windows(t_files[idx], config)
            needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
            if needs_raw:
                X_raw, _, _ = preprocess_subject_windows(t_files[idx], config, apply_filter=False)
            print(f"Trials: {X.shape[0]}  Shape: {X.shape}")

            for name in args.models:
                display = MODEL_DISPLAY[name]
                print(f"\n{'=' * 50}")
                print(f"  Running: {display}")
                print(f"{'=' * 50}")
                X_use = X_raw if name in MODELS_NO_FILTER else X
                acc, kappa, t = run_subject_cv(name, X_use, y, config,
                                               n_splits=args.n_splits, augment=False)
                print(f"  {display} CV mean: {acc*100:.2f}%  kappa: {kappa:.2f}")
                results[display] = (acc, 0.0, kappa, 0.0, t)

            data_label = f"subject {args.subject} ({args.n_splits}-fold CV)"

    # ── Dataset 2a ────────────────────────────────────────────────────────
    else:
        t_files = get_training_files_2a("data/2a")
        e_files = get_evaluation_files_2a("data/2a")

        # ── T-E holdout ───────────────────────────────────────────────────
        if args.protocol == "te":
            if args.all_subjects:
                print(f"\nRunning T-E holdout on all {len(t_files)} subjects...")
                # Clear checkpoint file so this run starts fresh
                open(f"results_{args.dataset}_{args.protocol}_checkpoint.txt", "w", encoding="utf-8").close()
                subj_accs   = {n: [] for n in args.models}
                subj_kappas = {n: [] for n in args.models}
                subj_times  = {n: [] for n in args.models}
                per_subject = {MODEL_DISPLAY[n]: [] for n in args.models}

                for i, (tf, ef) in enumerate(zip(t_files, e_files), start=1):
                    print(f"\n--- Subject {i} ---")
                    X_tr, y_tr, _ = preprocess_subject_windows_2a(tf, config)
                    X_te, y_te, _ = preprocess_subject_windows_2a(ef, config)
                    needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                    if needs_raw:
                        X_tr_raw, _, _ = preprocess_subject_windows_2a(tf, config, apply_filter=False)
                        X_te_raw, _, _ = preprocess_subject_windows_2a(ef, config, apply_filter=False)
                    # MCSANet and ATCNet use B=3 (1.5-6.0s, 1125 samples) to match their papers
                    if "mcsanet" in args.models or "atcnet" in args.models:
                        config_b3 = PreprocessingConfig(A=config.A, B=3, C=config.C, D=config.D)
                        X_tr_mcsanet, _, _ = preprocess_subject_windows_2a(tf, config_b3, apply_filter=False)
                        X_te_mcsanet, _, _ = preprocess_subject_windows_2a(ef, config_b3, apply_filter=False)

                    for name in args.models:
                        display = MODEL_DISPLAY[name]
                        print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                        if name in ("mcsanet", "atcnet"):
                            Xtr, Xte = X_tr_mcsanet, X_te_mcsanet
                        else:
                            Xtr = X_tr_raw if name in MODELS_NO_FILTER else X_tr
                            Xte = X_te_raw if name in MODELS_NO_FILTER else X_te
                        acc, kappa, t = run_model_holdout(name, Xtr, y_tr, Xte, y_te, config, augment=True)
                        print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                        subj_accs[name].append(acc)
                        subj_kappas[name].append(kappa)
                        subj_times[name].append(t)
                        per_subject[display].append((acc, kappa))
                        save_partial_result(args.dataset, args.protocol, i-1, display, acc, kappa)

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    results[display] = (
                        np.mean(subj_accs[name]),
                        np.std(subj_accs[name]),
                        np.mean(subj_kappas[name]),
                        np.std(subj_kappas[name]),
                        np.mean(subj_times[name]),
                    )
                data_label = "all subjects (T-E holdout)"

            else:
                idx = args.subject - 1
                print(f"\nLoading subject {args.subject} (T-E holdout)...")
                X_tr, y_tr, _ = preprocess_subject_windows_2a(t_files[idx], config)
                X_te, y_te, _ = preprocess_subject_windows_2a(e_files[idx], config)
                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_tr_raw, _, _ = preprocess_subject_windows_2a(t_files[idx], config, apply_filter=False)
                    X_te_raw, _, _ = preprocess_subject_windows_2a(e_files[idx], config, apply_filter=False)
                # MCSANet uses B=3 (1.5-6.0s, 1125 samples) to match paper
                if "mcsanet" in args.models:
                    config_b3 = PreprocessingConfig(A=config.A, B=3, C=config.C, D=config.D)
                    X_tr_mcsanet, _, _ = preprocess_subject_windows_2a(t_files[idx], config_b3, apply_filter=False)
                    X_te_mcsanet, _, _ = preprocess_subject_windows_2a(e_files[idx], config_b3, apply_filter=False)
                print(f"Train trials: {X_tr.shape[0]}  Test trials: {X_te.shape[0]}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                    if name == "mcsanet":
                        Xtr, Xte = X_tr_mcsanet, X_te_mcsanet
                    else:
                        Xtr = X_tr_raw if name in MODELS_NO_FILTER else X_tr
                        Xte = X_te_raw if name in MODELS_NO_FILTER else X_te
                    acc, kappa, t = run_model_holdout(name, Xtr, y_tr, Xte, y_te, config, augment=True)
                    print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    results[display] = (acc, 0.0, kappa, 0.0, t)
                data_label = f"subject {args.subject} (T-E holdout)"

        # ── LOSO ─────────────────────────────────────────────────────────
        elif args.protocol == "loso":
            loso_indices = list(range(len(t_files))) if args.all_subjects else [args.subject - 1]
            if args.all_subjects:
                print(f"\nRunning LOSO on all {len(t_files)} subjects (2a)...")
                completed, saved = load_completed_results(args.dataset, args.protocol)
                if completed:
                    print(f"  Resuming — {len(completed)} results already in checkpoint, skipping those.")
                else:
                    open(f"results_{args.dataset}_{args.protocol}_checkpoint.txt", "w", encoding="utf-8").close()
                    saved = {}
            else:
                print(f"\nRunning LOSO fold for subject {args.subject} (2a)...")
                completed, saved = set(), {}
            subj_accs   = {n: [] for n in args.models}
            subj_kappas = {n: [] for n in args.models}
            subj_times  = {n: [] for n in args.models}
            per_subject = {MODEL_DISPLAY[n]: [] for n in args.models}
            # Pre-populate results from checkpoint
            for (sidx, mdisplay), (acc, kappa) in saved.items():
                for n in args.models:
                    if MODEL_DISPLAY[n] == mdisplay:
                        subj_accs[n].append(acc)
                        subj_kappas[n].append(kappa)
                        subj_times[n].append(0.0)
                        per_subject[mdisplay].append((acc, kappa))

            for test_idx in loso_indices:
                print(f"\n--- LOSO: Test subject {test_idx+1} ---")
                train_files = [f for i, f in enumerate(t_files) if i != test_idx]

                X_tr_parts, y_tr_parts = [], []
                for f in train_files:
                    X_, y_, _ = preprocess_subject_windows_2a(f, config)
                    X_tr_parts.append(X_)
                    y_tr_parts.append(y_)
                X_train = np.concatenate(X_tr_parts, axis=0)
                y_train = np.concatenate(y_tr_parts, axis=0)

                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_tr_raw_parts = []
                    for f in train_files:
                        X_raw_, _, _ = preprocess_subject_windows_2a(f, config, apply_filter=False)
                        X_tr_raw_parts.append(X_raw_)
                    X_train_raw = np.concatenate(X_tr_raw_parts, axis=0)

                X_test, y_test, _ = preprocess_subject_windows_2a(e_files[test_idx], config)
                if needs_raw:
                    X_test_raw, _, _ = preprocess_subject_windows_2a(e_files[test_idx], config, apply_filter=False)

                # MCSANet and ATCNet use B=3 (1.5-6.0s, 1125 samples) to match their papers
                if "mcsanet" in args.models or "atcnet" in args.models:
                    config_b3 = PreprocessingConfig(A=config.A, B=3, C=config.C, D=config.D)
                    X_tr_mcsanet_parts = []
                    for f in train_files:
                        X_m_, _, _ = preprocess_subject_windows_2a(f, config_b3, apply_filter=False)
                        X_tr_mcsanet_parts.append(X_m_)
                    X_train_mcsanet = np.concatenate(X_tr_mcsanet_parts, axis=0)
                    X_test_mcsanet, _, _ = preprocess_subject_windows_2a(e_files[test_idx], config_b3, apply_filter=False)

                print(f"  Train: {X_train.shape[0]} trials  Test: {X_test.shape[0]} trials")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    if (test_idx, display) in completed:
                        print(f"  Skipping {display} (already done)")
                        continue
                    print(f"\n{'=' * 50}\n  Running: {display}\n{'=' * 50}")
                    if name in ("mcsanet", "atcnet"):
                        Xtr, Xte = X_train_mcsanet, X_test_mcsanet
                    else:
                        Xtr = X_train_raw if name in MODELS_NO_FILTER else X_train
                        Xte = X_test_raw if name in MODELS_NO_FILTER else X_test
                    acc, kappa, t = run_model_holdout(name, Xtr, y_train, Xte, y_test, config, augment=True)
                    print(f"  {display}: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    subj_accs[name].append(acc)
                    subj_kappas[name].append(kappa)
                    subj_times[name].append(t)
                    per_subject[display].append((acc, kappa))
                    save_partial_result(args.dataset, args.protocol, test_idx, display, acc, kappa)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_kappas[name]),
                    np.std(subj_kappas[name]),
                    np.mean(subj_times[name]),
                )
            data_label = f"subject {args.subject} (LOSO fold)" if not args.all_subjects else "all subjects (LOSO)"

        # ── 10-fold CV ────────────────────────────────────────────────────
        elif args.all_subjects:
            print(f"\nRunning {args.n_splits}-fold CV on all {len(t_files)} subjects...")
            subj_accs   = {n: [] for n in args.models}
            subj_kappas = {n: [] for n in args.models}
            subj_times  = {n: [] for n in args.models}

            for i, tf in enumerate(t_files, start=1):
                print(f"\n--- Subject {i} ---")
                X, y, _ = preprocess_subject_windows_2a(tf, config)
                needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
                if needs_raw:
                    X_raw, _, _ = preprocess_subject_windows_2a(
                        tf, config, apply_filter=False)
                print(f"  Trials: {X.shape[0]}  Shape: {X.shape}")

                for name in args.models:
                    display = MODEL_DISPLAY[name]
                    print(f"\n{'=' * 50}")
                    print(f"  Running: {display}")
                    print(f"{'=' * 50}")
                    X_use = X_raw if name in MODELS_NO_FILTER else X
                    acc, kappa, t = run_subject_cv(name, X_use, y, config,
                                                   n_splits=args.n_splits)
                    print(f"  {display} CV mean: {acc*100:.2f}%  kappa: {kappa:.2f}")
                    subj_accs[name].append(acc)
                    subj_kappas[name].append(kappa)
                    subj_times[name].append(t)

            for name in args.models:
                display = MODEL_DISPLAY[name]
                results[display] = (
                    np.mean(subj_accs[name]),
                    np.std(subj_accs[name]),
                    np.mean(subj_kappas[name]),
                    np.std(subj_kappas[name]),
                    np.mean(subj_times[name]),
                )
            data_label = f"all subjects ({args.n_splits}-fold CV)"

        else:
            idx = args.subject - 1
            if idx < 0 or idx >= len(t_files):
                raise ValueError(f"Subject {args.subject} out of range (1-{len(t_files)}).")
            print(f"\nLoading subject {args.subject} ...")
            X, y, _ = preprocess_subject_windows_2a(t_files[idx], config)
            needs_raw = any(n in MODELS_NO_FILTER for n in args.models)
            if needs_raw:
                X_raw, _, _ = preprocess_subject_windows_2a(
                    t_files[idx], config, apply_filter=False)
            print(f"Trials: {X.shape[0]}  Shape: {X.shape}")

            for name in args.models:
                display = MODEL_DISPLAY[name]
                print(f"\n{'=' * 50}")
                print(f"  Running: {display}")
                print(f"{'=' * 50}")
                X_use = X_raw if name in MODELS_NO_FILTER else X
                acc, kappa, t = run_subject_cv(name, X_use, y, config,
                                               n_splits=args.n_splits)
                print(f"  {display} CV mean: {acc*100:.2f}%  kappa: {kappa:.2f}")
                results[display] = (acc, 0.0, kappa, 0.0, t)

            data_label = f"subject {args.subject} ({args.n_splits}-fold CV)"

    total_time = time.time() - total_start

    print(f"\nData: {data_label}")
    print_results_table(results, per_subject=per_subject)
    print(f"\nTotal experiment time: {total_time:.1f} s")
    save_results_to_file(results, data_label, args.dataset, args.protocol,
                         total_time, per_subject=per_subject)


if __name__ == "__main__":
    main()
