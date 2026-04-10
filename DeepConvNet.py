import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from EEGModels import DeepConvNet
from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)
from cross_validation import make_cv_splits


def prepare_deepconvnet_input(X, y):
    """
    Convert preprocessing output to DeepConvNet input.

    preprocessing.py returns:
        X shape = (n_samples, n_times, n_channels)

    DeepConvNet Keras expects:
        X shape = (n_samples, n_channels, n_times, 1)

    Also convert labels from {1,2} -> {0,1}
    """
    X = np.transpose(X, (0, 2, 1))   # (N, T, C) -> (N, C, T)
    X = X[..., np.newaxis]           # (N, C, T) -> (N, C, T, 1)

    y = y.astype(int) - 1

    return X.astype(np.float32), y.astype(np.int64)


def run_deepconvnet_cv(
    X,
    y,
    groups,
    config,
    n_splits=10,
    epochs=50,
    batch_size=16,
    learning_rate=5e-4,
):
    """
    Run DeepConvNet with cross-validation using the same split logic
    as the ML baseline.
    """
    X, y = prepare_deepconvnet_input(X, y)

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running DeepConvNet =====")
    print("Prepared X shape:", X.shape)
    print("Prepared y shape:", y.shape)

    n_classes = len(np.unique(y))
    n_channels = X.shape[1]
    n_samples = X.shape[2]

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = DeepConvNet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_samples,
            dropoutRate=0.5,
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0,
        )

        end = time.time()

        train_time = end - start
        times.append(train_time)

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"DeepConvNet Fold {fold} accuracy: {acc:.4f}")
        print(f"DeepConvNet Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


if __name__ == "__main__":
    files = get_training_files("data")

    # your chosen preprocessing factors
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running DeepConvNet experiment with config:", config)

    # start with one subject first
    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("groups is None:", groups is None)
    if groups is not None:
        print("groups shape:", groups.shape)

    deepconv_accuracies, deepconv_times = run_deepconvnet_cv(
        X,
        y,
        groups,
        config=config,
        n_splits=10,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
    )

    print("\n===== DeepConvNet Final Results =====")
    print(f"Mean accuracy: {np.mean(deepconv_accuracies):.4f}")
    print(f"Std accuracy : {np.std(deepconv_accuracies):.4f}")
    print(f"Avg training time: {np.mean(deepconv_times):.4f} s")