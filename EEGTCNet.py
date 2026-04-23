import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Permute, Conv2D, Conv1D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D, Dropout,
    Dense, Lambda, Add
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)
from cross_validation import make_cv_splits


def prepare_input(X, y):
    """
    preprocessing.py output:
        X: (N, T, C)

    Convert to:
        X: (N, 1, C, T)
    """
    X = np.transpose(X, (0, 2, 1))   # (N, C, T)
    X = X[:, np.newaxis, :, :]       # (N, 1, C, T)
    y = y.astype(int) - 1            # {1,2} -> {0,1}
    return X.astype(np.float32), y.astype(np.int64)


def eegnet_backbone(input_layer, F1=8, kernLength=32, D=2, Chans=3, dropout=0.2):
    """
    EEGNet feature extractor used inside EEGTCNet.
    Input shape after permute: (None, T, C, 1)  channels_last
    """
    F2 = F1 * D

    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(input_layer)
    x = BatchNormalization(axis=-1)(x)

    x = DepthwiseConv2D(
        (1, Chans),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.)
    )(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropout)(x)

    x = SeparableConv2D(F2, (16, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropout)(x)

    return x


def tcn_block(input_layer, input_dimension, depth=2, kernel_size=4, filters=12,
              dropout=0.3, activation='elu'):
    """
    Simple TCN block adapted from the repo.
    Input shape: (None, seq_len, features)
    """
    x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1,
               activation='linear', padding='causal',
               kernel_initializer='he_uniform')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1,
               activation='linear', padding='causal',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)

    if input_dimension != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        out = Add()([x, shortcut])
    else:
        out = Add()([x, input_layer])

    out = Activation(activation)(out)

    for i in range(depth - 1):
        x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                   activation='linear', padding='causal',
                   kernel_initializer='he_uniform')(out)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                   activation='linear', padding='causal',
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        out = Add()([x, out])
        out = Activation(activation)(out)

    return out


def EEGTCNet(n_classes, Chans=3, Samples=1000, layers=2, kernel_s=4, filt=12,
             dropout=0.3, activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=0.2):
    """
    Adapted EEGTCNet for BCI IV 2b.
    Input shape: (None, 1, Chans, Samples)
    """
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)   # -> (None, Samples, Chans, 1)

    regRate = 0.25
    F2 = F1 * D

    feat = eegnet_backbone(
        input_layer=input2,
        F1=F1,
        kernLength=kernLength,
        D=D,
        Chans=Chans,
        dropout=dropout_eeg
    )

    feat = Lambda(lambda x: x[:, :, -1, :])(feat)   # -> (None, seq_len, F2)
    tcn_out = tcn_block(
        input_layer=feat,
        input_dimension=F2,
        depth=layers,
        kernel_size=kernel_s,
        filters=filt,
        dropout=dropout,
        activation=activation
    )

    out = Lambda(lambda x: x[:, -1, :])(tcn_out)
    out = Dense(n_classes, kernel_constraint=max_norm(regRate))(out)
    out = Activation('softmax')(out)

    return Model(inputs=input1, outputs=out)


def run_eegtcnet_cv(X, y, groups, config, n_splits=10, epochs=80, batch_size=16, learning_rate=5e-4):
    X, y = prepare_input(X, y)
    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running EEGTCNet =====")
    print("Prepared X shape:", X.shape)
    print("Prepared y shape:", y.shape)

    n_classes = len(np.unique(y))
    n_channels = X.shape[2]
    n_samples = X.shape[3]

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = EEGTCNet(
            n_classes=n_classes,
            Chans=n_channels,
            Samples=n_samples,
            layers=2,
            kernel_s=4,
            filt=12,
            dropout=0.3,
            activation='elu',
            F1=8,
            D=2,
            kernLength=32,
            dropout_eeg=0.2
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=12,
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

        print(f"EEGTCNet Fold {fold} accuracy: {acc:.4f}")
        print(f"EEGTCNet Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


def run_eegtcnet_holdout(X_train, y_train, X_test, y_test,
                         epochs=80, batch_size=16, learning_rate=5e-4):
    """Train on X_train, evaluate on X_test (T→E protocol)."""
    X_train, y_train = prepare_input(X_train, y_train)
    X_test, y_test = prepare_input(X_test, y_test)

    n_classes = len(np.unique(y_train))
    n_channels = X_train.shape[2]
    n_samples = X_train.shape[3]

    model = EEGTCNet(n_classes=n_classes, Chans=n_channels, Samples=n_samples,
                     layers=2, kernel_s=4, filt=12, dropout=0.3,
                     activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=0.2)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=12,
                               restore_best_weights=True, verbose=0)
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[early_stop], verbose=0)
    end = time.time()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"EEGTCNet holdout accuracy: {acc:.4f}  time: {end-start:.1f}s")
    return acc, end - start


if __name__ == "__main__":
    files = get_training_files("data/2b")
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running EEGTCNet experiment with config:", config)

    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("groups is None:", groups is None)

    accs, ts = run_eegtcnet_cv(
        X, y, groups,
        config=config,
        n_splits=10,
        epochs=80,
        batch_size=16,
        learning_rate=1e-3,
    )

    print("\n===== EEGTCNet Final Results =====")
    print(f"Mean accuracy: {np.mean(accs):.4f}")
    print(f"Std accuracy : {np.std(accs):.4f}")
    print(f"Avg training time: {np.mean(ts):.4f} s")