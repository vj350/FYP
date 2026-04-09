import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)


def make_cv_splits(X, y, config: PreprocessingConfig, groups=None, n_splits=10):
    """
    Create cross-validation splits based on factor A.

    Rule:
    - A1 -> KFold
    - A2/A3/A4 -> GroupKFold

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    config : PreprocessingConfig
    groups : np.ndarray | None
    n_splits : int

    Returns
    -------
    splits : list of (train_idx, test_idx)
    """
    splits = []

    if config.A == 1:
        cv = KFold(n_splits=n_splits, shuffle=False)
        for train_idx, test_idx in cv.split(X, y):
            splits.append((train_idx, test_idx))
    else:
        if groups is None:
            raise ValueError(
                "groups cannot be None when A is 2, 3, or 4. "
                "Windowed data requires GroupKFold to avoid leakage."
            )

        cv = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in cv.split(X, y, groups):
            splits.append((train_idx, test_idx))

    return splits


if __name__ == "__main__":
    files = get_training_files("data")

    # Change A here to test KFold vs GroupKFold behavior
    config = PreprocessingConfig(A=4, B=1, C=1, D=1)

    X, y, groups = preprocess_subject_windows(files[0], config)

    print("Dataset loaded for CV:")
    print("X shape     :", X.shape)
    print("y shape     :", y.shape)
    print("groups is None:", groups is None)
    if groups is not None:
        print("groups shape:", groups.shape)
        print("Unique groups:", len(np.unique(groups)))
    print()

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=10)

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"Fold {i}")
        print("  Train samples:", len(train_idx))
        print("  Test samples :", len(test_idx))

        if config.A == 1:
            print("  CV method    : KFold")
        else:
            train_groups = np.unique(groups[train_idx])
            test_groups = np.unique(groups[test_idx])
            overlap = np.intersect1d(train_groups, test_groups)

            print("  CV method    : GroupKFold")
            print("  Train groups :", len(train_groups))
            print("  Test groups  :", len(test_groups))
            print("  Group overlap:", len(overlap))

        print()