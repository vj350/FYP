import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from EEGModels import ShallowConvNet
from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)
from cross_validation import make_cv_splits


def prepare_shallowconvnet_input(X, y):
    """
    Convert preprocessing output to ShallowConvNet input.

    preprocessing.py returns:
        X shape = (n_samples, n_times, n_channels)

    ShallowConvNet Keras expects:
        X shape = (n_samples, n_channels, n_times, 1)

    Also convert labels from {1,2} -> {0,1}
    """
    X = np.transpose(X, (0, 2, 1))   # (N, T, C) -> (N, C, T)
    X = X[..., np.newaxis]           # (N, C, T) -> (N, C, T, 1)

    y = y.astype(int) - 1

    return X.astype(np.float32), y.astype(np.int64)


def run_shallowconvnet_cv(
    X,
    y,
    groups,
    config,
    n_splits=10,
    epochs=50,
    batch_size=16,
    learning_rate=1e-3,
):
    """
    Run ShallowConvNet with cross-validation using the same split logic
    as the ML baseline.
    """
    X, y = prepare_shallowconvnet_input(X, y)

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running ShallowConvNet =====")
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

        model = ShallowConvNet(
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

        print(f"ShallowConvNet Fold {fold} accuracy: {acc:.4f}")
        print(f"ShallowConvNet Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


def run_shallowconvnet_holdout(X_train, y_train, X_test, y_test,
                               epochs=50, batch_size=16, learning_rate=1e-3):
    """Train on X_train, evaluate on X_test (T→E protocol)."""
    X_train, y_train = prepare_shallowconvnet_input(X_train, y_train)
    X_test, y_test = prepare_shallowconvnet_input(X_test, y_test)

    n_classes = len(np.unique(y_train))
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    model = ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_samples,
                           dropoutRate=0.5)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[early_stop], verbose=0)
    end = time.time()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"ShallowConvNet holdout accuracy: {acc:.4f}  time: {end-start:.1f}s")
    return acc, end - start


if __name__ == "__main__":
    files = get_training_files("data/2b")

    # your chosen preprocessing factors
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running ShallowConvNet experiment with config:", config)

    # start with one subject first
    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("groups is None:", groups is None)
    if groups is not None:
        print("groups shape:", groups.shape)

    shallowconv_accuracies, shallowconv_times = run_shallowconvnet_cv(
        X,
        y,
        groups,
        config=config,
        n_splits=10,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
    )

    print("\n===== ShallowConvNet Final Results =====")
    print(f"Mean accuracy: {np.mean(shallowconv_accuracies):.4f}")
    print(f"Std accuracy : {np.std(shallowconv_accuracies):.4f}")
    print(f"Avg training time: {np.mean(shallowconv_times):.4f} s")