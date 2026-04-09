from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt


# =========================
# Configuration
# =========================
FS_DEFAULT = 250
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 4

TRIAL_DURATION_SEC = 8.0      # Paper F prototype: full 0–8 s interval
WINDOW_SEC = 2.0
STEP_SEC = 0.1

N_EEG_CHANNELS = 3            # first 3 are EEG, last 3 are EOG
DROP_ARTIFACTS = True         # remove trials marked as artifacts


# =========================
# Basic signal processing
# =========================
def bandpass_filter(
    data: np.ndarray,
    fs: int = FS_DEFAULT,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Band-pass filter.
    Input shape: (samples, channels)
    Output shape: (samples, channels)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def sliding_window(
    trial_data: np.ndarray,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    fs: int = FS_DEFAULT,
) -> np.ndarray:
    """
    Create overlapping sliding windows from one trial.

    Input:
        trial_data shape: (samples, channels)

    Output:
        windows shape: (n_windows, window_samples, channels)
    """
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    n_samples, _ = trial_data.shape
    windows = []

    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(trial_data[start:end, :])

    return np.stack(windows, axis=0)


# =========================
# MATLAB file loading
# =========================
def load_subject_mat(file_path: str | Path) -> np.ndarray:
    """
    Load one subject .mat file.

    Returns:
        data object with shape (1, n_blocks)
    """
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise KeyError(f"'data' key not found in {file_path}")
    return mat["data"]


def _unwrap_block(block_struct: np.ndarray):
    """
    Unwrap one MATLAB structured block.
    """
    return block_struct[0, 0]


# =========================
# Trial extraction
# =========================
def extract_trials_from_block(
    block_struct,
    trial_duration_sec=TRIAL_DURATION_SEC,
):
    """
    Extract clean trials from one block.
    Artifact trials are removed.
    """

    block = block_struct[0, 0]

    X = block["X"]                     # continuous signal
    trial_starts = block["trial"].flatten()
    labels = block["y"].flatten()
    artifacts = block["artifacts"].flatten()
    fs_value = int(block["fs"].flatten()[0])

    # keep only EEG channels (C3, Cz, C4)
    X_eeg = X[:, :3]

    trial_samples = int(trial_duration_sec * fs_value)

    trials = []
    y_trials = []

    for start, label, artifact in zip(trial_starts, labels, artifacts):

        # 🚨 remove artifact trials
        if artifact == 1:
            continue

        # MATLAB indexing starts at 1
        start_idx = int(start) - 1
        end_idx = start_idx + trial_samples

        if end_idx > X_eeg.shape[0]:
            continue

        trial_data = X_eeg[start_idx:end_idx, :]

        trials.append(trial_data)
        y_trials.append(int(label))

    return trials, y_trials

def preprocess_subject_trials(
    file_path: str | Path,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load one subject file and return full filtered trials.

    Returns:
        trials_filtered : list of arrays, each shape (trial_samples, 3)
        labels          : np.ndarray shape (n_trials,)
    """
    data = load_subject_mat(file_path)

    all_trials = []
    all_labels = []

    n_blocks = data.shape[1]

    for i in range(n_blocks):
        block_struct = data[0, i]
        trials, labels = extract_trials_from_block(block_struct)

        # filter each full trial
        for trial in trials:
            filtered_trial = bandpass_filter(trial)
            all_trials.append(filtered_trial)

        all_labels.extend(labels)

    return all_trials, np.array(all_labels, dtype=int)


# =========================
# Windowed dataset creation
# =========================
def make_window_dataset(
    trials: List[np.ndarray],
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert full trials into windowed dataset.

    Returns:
        X_windows : shape (total_windows, window_samples, channels)
        y_windows : shape (total_windows,)
        groups    : shape (total_windows,)
            group id = original trial id
            use this for leakage-safe cross-validation
    """
    X_all = []
    y_all = []
    groups = []

    for trial_id, (trial, label) in enumerate(zip(trials, labels)):
        windows = sliding_window(trial)   # (n_windows, 500, 3)
        n_windows = windows.shape[0]

        X_all.append(windows)
        y_all.append(np.full(n_windows, label, dtype=int))
        groups.append(np.full(n_windows, trial_id, dtype=int))

    X_windows = np.concatenate(X_all, axis=0)
    y_windows = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups, axis=0)

    return X_windows, y_windows, groups


# =========================
# Full file-level pipeline
# =========================
def preprocess_subject_windows(
    file_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline for one subject file:
        load -> extract trials -> keep EEG only -> band-pass -> window -> labels/groups

    Returns:
        X : (total_windows, 500, 3)
        y : (total_windows,)
        groups : (total_windows,)
    """
    trials, labels = preprocess_subject_trials(file_path)
    return make_window_dataset(trials, labels)


def preprocess_multiple_subjects(
    file_paths: List[str | Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess multiple subject files and combine them.

    Returns:
        X : (total_windows, window_samples, channels)
        y : (total_windows,)
        groups : (total_windows,)
            globally unique trial ids across all subjects
    """
    X_list = []
    y_list = []
    groups_list = []

    trial_offset = 0

    for file_path in file_paths:
        X_subj, y_subj, groups_subj = preprocess_subject_windows(file_path)

        # make group ids globally unique across subjects
        groups_subj = groups_subj + trial_offset
        trial_offset = groups_subj.max() + 1

        X_list.append(X_subj)
        y_list.append(y_subj)
        groups_list.append(groups_subj)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)

    return X, y, groups


# =========================
# Helper to load all training files
# =========================
def get_training_files(data_dir: str | Path) -> List[Path]:
    """
    Return all B??T.mat files in sorted order.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("B??T.mat"))
    if not files:
        raise FileNotFoundError(f"No training files found in {data_dir}")
    return files


# =========================
# Quick test
# =========================
if __name__ == "__main__":
    data_dir = Path("data")

    files = get_training_files(data_dir)
    print("Training files found:")
    for f in files:
        print(" ", f.name)

    # Test on first subject only
    X, y, groups = preprocess_subject_windows(files[0])

    print("\nOne-subject windowed dataset:")
    print("X shape     :", X.shape)
    print("y shape     :", y.shape)
    print("groups shape:", groups.shape)
    print("Unique labels:", np.unique(y))
    print("First window shape:", X[0].shape)