import numpy as np
from sklearn.model_selection import GroupKFold

from preprocessing import get_training_files, preprocess_subject_windows


def make_group_cv_splits(X, y, groups, n_splits=10):
    """
    Group-based cross-validation.
    Keeps all windows from the same original trial in the same fold.
    """
    gkf = GroupKFold(n_splits=n_splits)

    splits = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        splits.append((train_idx, test_idx))

    return splits


if __name__ == "__main__":
    files = get_training_files("data")

    # test on one subject first
    X, y, groups = preprocess_subject_windows(files[0])

    print("Dataset loaded for CV:")
    print("X shape     :", X.shape)
    print("y shape     :", y.shape)
    print("groups shape:", groups.shape)
    print("Unique groups:", len(np.unique(groups)))
    print()

    splits = make_group_cv_splits(X, y, groups, n_splits=10)

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        overlap = np.intersect1d(train_groups, test_groups)

        print(f"Fold {i}")
        print("  Train windows       :", len(train_idx))
        print("  Test windows        :", len(test_idx))
        print("  Unique train trials :", len(train_groups))
        print("  Unique test trials  :", len(test_groups))
        print("  Group overlap       :", len(overlap))
        print()