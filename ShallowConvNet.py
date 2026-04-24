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

    print("\nUse run_all.py to run experiments with the T-E holdout protocol.")