from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt


# =========================
# Constants
# =========================
FS_DEFAULT = 250
FILTER_ORDER = 4
N_EEG_CHANNELS = 3  # first 3 are EEG, last 3 are EOG


# =========================
# Config
# =========================
@dataclass
class PreprocessingConfig:
    """
    Paper F factor configuration.

    A: time window / step size
    B: time interval
    C: theta option
    D: mu/beta option
    """
    A: int = 4
    B: int = 1
    C: int = 1
    D: int = 1


# =========================
# Factor mapping helpers
# =========================
def get_window_params(A: int) -> Tuple[bool, float | None, float | None]:
    """
    Map factor A to windowing behavior.

    Returns
    -------
    use_windowing : bool
    window_sec : float | None
    step_sec : float | None
    """
    if A == 1:
        return False, None, None
    elif A == 2:
        return True, 1.0, 1.0
    elif A == 3:
        return True, 2.0, 2.0
    elif A == 4:
        return True, 2.0, 0.1
    else:
        raise ValueError(f"Invalid A level: {A}. Must be 1, 2, 3, or 4.")


def get_time_interval(B: int) -> Tuple[float, float]:
    """
    Map factor B to trial time interval in seconds.

    Returns
    -------
    start_sec, end_sec
    """
    if B == 1:
        return 0.0, 8.0
    elif B == 2:
        return 3.5, 7.5
    else:
        raise ValueError(f"Invalid B level: {B}. Must be 1 or 2.")


def get_bandpass_range(C: int, D: int) -> Tuple[float, float]:
    """
    Map C and D to one broad preprocessing band-pass range.

    C1 = no theta band
    C2 = include theta band (4-8 Hz)

    D1 = 8-30 Hz
    D2 = 8-40 Hz

    Combined behavior:
    - C1, D1 -> 8-30 Hz
    - C2, D1 -> 4-30 Hz
    - C1, D2 -> 8-40 Hz
    - C2, D2 -> 4-40 Hz
    """
    if C == 1:
        lowcut = 8.0
    elif C == 2:
        lowcut = 4.0
    else:
        raise ValueError(f"Invalid C level: {C}. Must be 1 or 2.")

    if D == 1:
        highcut = 30.0
    elif D == 2:
        highcut = 40.0
    else:
        raise ValueError(f"Invalid D level: {D}. Must be 1 or 2.")

    return lowcut, highcut


# =========================
# Basic signal processing
# =========================
def bandpass_filter(
    data: np.ndarray,
    fs: int = FS_DEFAULT,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Band-pass filter.

    Input shape:
        (samples, channels)

    Output shape:
        (samples, channels)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def sliding_window(
    trial_data: np.ndarray,
    window_sec: float,
    step_sec: float,
    fs: int = FS_DEFAULT,
) -> np.ndarray:
    """
    Create windows from one trial.

    Parameters
    ----------
    trial_data : np.ndarray
        Shape (samples, channels)

    Returns
    -------
    windows : np.ndarray
        Shape (n_windows, window_samples, channels)
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
# MATLAB loading
# =========================
def load_subject_mat(file_path: str | Path) -> np.ndarray:
    """
    Load one subject .mat file.

    Returns
    -------
    data : np.ndarray
        MATLAB data object with shape (1, n_blocks)
    """
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise KeyError(f"'data' key not found in {file_path}")
    return mat["data"]


# =========================
# Trial extraction
# =========================
def extract_trials_from_block(
    block_struct,
    config: PreprocessingConfig,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Extract clean trials from one block using factor B time interval.

    This function does:
    - keep EEG channels only
    - remove artifact trials
    - extract selected time interval
    """
    block = block_struct[0, 0]

    X = block["X"]
    trial_starts = block["trial"].flatten()
    labels = block["y"].flatten()
    artifacts = block["artifacts"].flatten()
    fs_value = int(block["fs"].flatten()[0])

    # keep EEG only: C3, Cz, C4
    X_eeg = X[:, :N_EEG_CHANNELS]

    start_sec, end_sec = get_time_interval(config.B)
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


def preprocess_subject_trials(
    file_path: str | Path,
    config: PreprocessingConfig,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Load one subject file and return extracted + filtered full trials.

    Preprocessing uses one broad band-pass determined by C and D:
    - C1/D1 -> 8-30
    - C2/D1 -> 4-30
    - C1/D2 -> 8-40
    - C2/D2 -> 4-40
    """
    data = load_subject_mat(file_path)

    all_trials = []
    all_labels = []
    fs_value = FS_DEFAULT

    lowcut, highcut = get_bandpass_range(config.C, config.D)

    n_blocks = data.shape[1]

    for i in range(n_blocks):
        block_struct = data[0, i]
        trials, labels, fs_value = extract_trials_from_block(block_struct, config)

        filtered_trials = [
            bandpass_filter(trial, fs=fs_value, lowcut=lowcut, highcut=highcut)
            for trial in trials
        ]

        all_trials.extend(filtered_trials)
        all_labels.extend(labels.tolist())

    return all_trials, np.array(all_labels, dtype=int), fs_value


# =========================
# Dataset creation
# =========================
def make_window_dataset(
    trials: List[np.ndarray],
    labels: np.ndarray,
    config: PreprocessingConfig,
    fs: int = FS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Convert trials into samples according to factor A.

    Returns
    -------
    X : np.ndarray
        Shape (total_samples, sample_len, channels)
    y : np.ndarray
        Shape (total_samples,)
    groups : np.ndarray | None
        For A1:
            returns None
        For A2/A3/A4:
            returns trial IDs for GroupKFold
    """
    X_all = []
    y_all = []
    groups = []

    use_windowing, window_sec, step_sec = get_window_params(config.A)

    for trial_id, (trial, label) in enumerate(zip(trials, labels)):
        if use_windowing:
            samples = sliding_window(
                trial_data=trial,
                window_sec=window_sec,
                step_sec=step_sec,
                fs=fs,
            )
            n_samples = samples.shape[0]

            X_all.append(samples)
            y_all.append(np.full(n_samples, label, dtype=int))
            groups.append(np.full(n_samples, trial_id, dtype=int))
        else:
            sample = trial[np.newaxis, :, :]
            X_all.append(sample)
            y_all.append(np.array([label], dtype=int))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    if use_windowing:
        groups = np.concatenate(groups, axis=0)
    else:
        groups = None

    return X, y, groups


# =========================
# Full pipelines
# =========================
def preprocess_subject_windows(
    file_path: str | Path,
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Full preprocessing for one subject file.

    Steps:
    - load file
    - extract trials using factor B
    - apply broad preprocessing band-pass using C/D
    - create samples using factor A
    """
    trials, labels, fs_value = preprocess_subject_trials(file_path, config)
    return make_window_dataset(trials, labels, config=config, fs=fs_value)


def preprocess_multiple_subjects(
    file_paths: List[str | Path],
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Preprocess multiple subjects and combine them.

    For A1:
        returns groups = None
    For A2/A3/A4:
        returns globally unique trial IDs for GroupKFold
    """
    X_list = []
    y_list = []
    groups_list = []

    use_windowing, _, _ = get_window_params(config.A)
    trial_offset = 0

    for file_path in file_paths:
        X_subj, y_subj, groups_subj = preprocess_subject_windows(file_path, config)

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


# =========================
# Helpers
# =========================
def get_training_files(data_dir: str | Path = "data/2b") -> List[Path]:
    """
    Return all B??T.mat files in sorted order.
    Default points to data/2b/ (Dataset 2b).
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
    data_dir = Path("data/2b")
    files = get_training_files(data_dir)

    config = PreprocessingConfig(A=4, B=1, C=1, D=1)

    print("Training files found:")
    for f in files:
        print(" ", f.name)

    print("\nConfig:")
    print(config)

    print("\nWindow params:", get_window_params(config.A))
    print("Time interval:", get_time_interval(config.B))
    print("Band-pass range:", get_bandpass_range(config.C, config.D))

    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nOne-subject dataset:")
    print("X shape     :", X.shape)
    print("y shape     :", y.shape)
    print("groups type :", type(groups))
    if groups is not None:
        print("groups shape:", groups.shape)
        print("Unique groups:", len(np.unique(groups)))
    print("Unique labels:", np.unique(y))
    print("First sample shape:", X[0].shape)