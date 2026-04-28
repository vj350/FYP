import time
import numpy as np

from scipy.signal import butter, filtfilt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, cohen_kappa_score

from EEGModels import ShallowConvNet
from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
    resample_to_128hz,
)

_FS      = 128
_WIN     = round(1.6 * _FS)        # 205 samples (1.6 s)
_STEP    = round(0.4 * _FS)        # 51  samples (0.4 s) — ~7x augmentation
_T_START = int(0.5 * _FS)          # 64  samples (0.5 s post-cue)
_T_END   = _T_START + _WIN          # 269 samples
_BASELINE = int(0.5 * _FS)         # 64  samples baseline


def _bandpass(X, fs=_FS, lowcut=1.0, highcut=50.0, order=4):
    """1-50 Hz 4th-order Butterworth bandpass (supervisor paper Section II.A). X: (N, C, T)"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, X, axis=2)


def _baseline_correct(X, n_baseline=_BASELINE):
    """Subtract per-trial per-channel mean of first n_baseline samples. X: (N, C, T)"""
    return X - X[:, :, :n_baseline].mean(axis=2, keepdims=True)


def prepare_shallowconvnet_input(X, y):
    """(N, T, C) -> (N, C, T, 1), labels {1,2,...} -> {0,1,...}"""
    X = np.transpose(X, (0, 2, 1))
    X = X[..., np.newaxis]
    y = y.astype(int) - 1
    return X.astype(np.float32), y.astype(np.int64)


def sliding_window_augment(X, y, window=_WIN, step=_STEP):
    """
    Sliding window augmentation at 128 Hz.
    window=205 (1.6 s), step=51 (0.4 s) -> ~7x augmentation.
    Matches supervisor paper (Section II.D).
    X: (N, T, C)
    """
    T = X.shape[1]
    X_aug, y_aug = [], []
    for i in range(X.shape[0]):
        for start in range(0, T - window + 1, step):
            X_aug.append(X[i, start:start + window, :])
            y_aug.append(y[i])
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)


def run_shallowconvnet_holdout(X_train, y_train, X_test, y_test,
                               epochs=200, batch_size=16, learning_rate=1e-3,
                               augment=True):
    """
    Train on X_train, evaluate on X_test.

    Preprocessing matches supervisor paper (Section II.A/D):
    - Downsample to 128 Hz
    - Baseline correction: subtract mean of first 0.5 s per trial
    - 1-50 Hz 4th-order Butterworth bandpass
    - augment=True:  sliding window 1.6 s / 0.4 s step (~7x) on train
    - augment=False: single [0.5, 2.1] s crop on train
    - Test always uses single [0.5, 2.1] s crop
    - 200 epochs with early stopping (patience=20)
    """
    X_train = resample_to_128hz(X_train)
    X_test  = resample_to_128hz(X_test)

    # (N, T, C) -> (N, C, T) for filtering
    X_train = np.transpose(X_train.astype(np.float32), (0, 2, 1))
    X_test  = np.transpose(X_test.astype(np.float32),  (0, 2, 1))

    X_train = _bandpass(_baseline_correct(X_train))
    X_test  = _bandpass(_baseline_correct(X_test))

    # (N, C, T) -> (N, T, C) for augmentation / cropping
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test  = np.transpose(X_test,  (0, 2, 1))

    if augment:
        X_train, y_train = sliding_window_augment(X_train, y_train)
    else:
        X_train = X_train[:, _T_START:_T_END, :]

    X_test = X_test[:, _T_START:_T_END, :]

    X_train, y_train = prepare_shallowconvnet_input(X_train, y_train)
    X_test,  y_test  = prepare_shallowconvnet_input(X_test,  y_test)

    n_classes  = len(np.unique(y_train))
    n_channels = X_train.shape[1]
    n_samples  = X_train.shape[2]

    model = ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_samples,
                           dropoutRate=0.5)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=20,
                               restore_best_weights=True, verbose=0)
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[early_stop], verbose=0)
    end = time.time()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_test, y_pred)
    kappa  = cohen_kappa_score(y_test, y_pred)
    print(f"ShallowConvNet accuracy: {acc:.4f}  kappa: {kappa:.2f}  time: {end-start:.1f}s")
    return acc, kappa, end - start


if __name__ == "__main__":
    files  = get_training_files("data/2b")
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)
    print("Use run_all.py to run experiments.")
