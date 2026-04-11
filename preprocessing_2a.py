from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

from preprocessing import (
    PreprocessingConfig,
    FS_DEFAULT,
    FILTER_ORDER,
    get_window_params,
    get_bandpass_range,
    bandpass_filter,
    sliding_window,
)


# =========================
# Constants
# =========================
N_EEG_CHANNELS_2A = 22   # first 22 are EEG, last 3 are EOG (25 total)
FIRST_TRIAL_BLOCK = 3    # blocks 0-2 are calibration (no trials), skip them


# =========================
# Factor B for Dataset 2a
# =========================
def get_time_interval_2a(B: int) -> Tuple[float, float]:
    """
    Map factor B to trial time interval for Dataset 2a.

    Based on the paradigm timing diagram:
      t=0-2s : fixation cross
      t=2-3s : cue
      t=3-6s : motor imagery
      t=6-8s : break

    B=1 : full trial          0.0 – 8.0s (2000 samples)
    B=2 : cue onset to end    2.0 – 6.0s (1000 samples)  [best: cue + full MI]
    B=3 : MCSANet window      1.5 – 6.0s (1125 samples)  [cue prep + full MI]
    B=4 : MI window only      3.0 – 6.0s  (750 samples)  [original, worse]
    """
    if B == 1:
        return 0.0, 8.0
    elif B == 2:
        return 2.0, 6.0
    elif B == 3:
        return 1.5, 6.0
    elif B == 4:
        return 3.0, 6.0
    else:
        raise ValueError(f"Invalid B level: {B}. Must be 1–4.")


# =========================
# Trial extraction
# =========================
def extract_trials_2a(
    block_struct,
    config: PreprocessingConfig,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Extract clean trials from one Dataset 2a block.

    - Takes first 22 channels (EEG only, discards 3 EOG)
    - Skips artifact trials
    - Extracts selected time interval (factor B)
    """
    block = block_struct[0, 0]

    X = block["X"]
    trial_starts = block["trial"].flatten()
    labels = block["y"].flatten()
    artifacts = block["artifacts"].flatten()
    fs_value = int(block["fs"].flatten()[0])

    # Skip blocks with no trial data (calibration blocks)
    if len(trial_starts) == 0:
        return [], np.array([], dtype=int), fs_value

    # Keep EEG channels only (first 22)
    X_eeg = X[:, :N_EEG_CHANNELS_2A]

    start_sec, end_sec = get_time_interval_2a(config.B)
    start_offset_samples = int(start_sec * fs_value)
    interval_samples = int((end_sec - start_sec) * fs_value)

    trials = []
    y_trials = []

    for start, label, artifact in zip(trial_starts, labels, artifacts):
        if artifact == 1:
            continue

        # MATLAB indexing starts at 1
        trial_start_idx = int(start) - 1

        start_idx = trial_start_idx + start_offset_samples
        end_idx = start_idx + interval_samples

        if end_idx > X_eeg.shape[0]:
            continue

        trial_data = X_eeg[start_idx:end_idx, :]
        trials.append(trial_data)
        y_trials.append(int(label))

    return trials, np.array(y_trials, dtype=int), fs_value


# =========================
# Full pipeline
# =========================
def preprocess_subject_trials_2a(
    file_path: str | Path,
    config: PreprocessingConfig,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Load one Dataset 2a subject file and return extracted + filtered trials.

    Same preprocessing method as Dataset 2b:
    - Bandpass filter determined by C and D factors
    - Time window determined by B factor (3-6s for B=2)
    - Artifact trials removed
    """
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise KeyError(f"'data' key not found in {file_path}")
    data = mat["data"]

    all_trials = []
    all_labels = []
    fs_value = FS_DEFAULT

    lowcut, highcut = get_bandpass_range(config.C, config.D)

    n_blocks = data.shape[1]

    for i in range(n_blocks):
        block_struct = data[0, i]
        trials, labels, fs_value = extract_trials_2a(block_struct, config)

        if len(trials) == 0:
            continue

        filtered_trials = [
            bandpass_filter(trial, fs=fs_value, lowcut=lowcut, highcut=highcut)
            for trial in trials
        ]

        all_trials.extend(filtered_trials)
        all_labels.extend(labels.tolist())

    return all_trials, np.array(all_labels, dtype=int), fs_value


def preprocess_subject_windows_2a(
    file_path: str | Path,
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Full preprocessing pipeline for one Dataset 2a subject file.

    Steps:
    - Load .mat file
    - Extract trials using factor B (3-6s for B=2)
    - Bandpass filter using C/D factors
    - Create samples using factor A

    Output format identical to Dataset 2b:
        X     : (N, T, C)  where C=22, T=750 for B=2
        y     : (N,)       labels in {1,2,3,4}
        groups: (N,) or None
    """
    from preprocessing import make_window_dataset
    trials, labels, fs_value = preprocess_subject_trials_2a(file_path, config)
    return make_window_dataset(trials, labels, config=config, fs=fs_value)


def preprocess_multiple_subjects_2a(
    file_paths: List[str | Path],
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Preprocess multiple Dataset 2a subjects and combine.
    """
    from preprocessing import get_window_params

    X_list = []
    y_list = []
    groups_list = []

    use_windowing, _, _ = get_window_params(config.A)
    trial_offset = 0

    for file_path in file_paths:
        X_subj, y_subj, groups_subj = preprocess_subject_windows_2a(file_path, config)

        X_list.append(X_subj)
        y_list.append(y_subj)

        if use_windowing and groups_subj is not None:
            groups_subj = groups_subj + trial_offset
            trial_offset = groups_subj.max() + 1
            groups_list.append(groups_subj)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if use_windowing:
        groups = np.concatenate(groups_list, axis=0)
    else:
        groups = None

    return X, y, groups


def get_training_files_2a(data_dir: str | Path = "data/2a") -> List[Path]:
    """
    Return all A??T.mat files in sorted order (Dataset 2a training files).
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("A??T.mat"))
    if not files:
        raise FileNotFoundError(f"No Dataset 2a training files found in {data_dir}")
    return files


def get_evaluation_files_2a(data_dir: str | Path = "data/2a") -> List[Path]:
    """
    Return all A??E.mat evaluation files in sorted order.
    Used for the official T→E evaluation protocol on Dataset 2a.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("A??E.mat"))
    if not files:
        raise FileNotFoundError(f"No Dataset 2a evaluation files found in {data_dir}")
    return files


# =========================
# Quick test
# =========================
if __name__ == "__main__":
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    files = get_training_files_2a()
    print(f"Found {len(files)} Dataset 2a subject files.")
    print(f"Config: {config}")
    print(f"Time interval: {get_time_interval_2a(config.B)}")

    X, y, groups = preprocess_subject_windows_2a(files[0], config)

    print(f"\nSubject 1:")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Unique labels: {np.unique(y)}")
    print(f"  groups  : {groups}")
