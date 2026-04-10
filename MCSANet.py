import time
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
    AveragePooling2D, Dropout, Dense, Flatten, Concatenate, Add,
    LayerNormalization, Lambda, Permute, Reshape
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.metrics import accuracy_score

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)
from cross_validation import make_cv_splits


def MCSANet(nb_classes, Chans=3, Samples=1000, F1=8, D=2, num_heads=2,
            dropout_rate=0.5, Fs=250):
    """
    Keras implementation of MCSANet:
    Multi-Scale Convolutional Self-Attention Network for MI-EEG classification.

    Reference:
        Devi et al. (2026). MCSANet: A multiscale convolutional self attention
        network with data augmentation for motor imagery EEG classification.
        Signal, Image and Video Processing, 20:190.

    NOTE FOR REPORT:
        The paper proposes two separate contributions:
          (1) MCSANet architecture — implemented here.
          (2) Temporal Segment Shuffling (TSS) — a data augmentation strategy
              that generates synthetic trials by recombining segments from 4
              trials of the same class. TSS is NOT part of the model architecture;
              it is applied to the training data before training.
        TSS is intentionally excluded here to ensure a fair comparison with the
        other models (EEGNet, DeepConvNet, ATCNet, etc.), none of which use
        data augmentation. The paper reports ~2.8% accuracy gain from TSS on
        Dataset 2b, so results here reflect the model architecture alone.

    Architecture:
        1. Convolutional Block:
           - Three parallel temporal Conv2D with kernels Fs/2, Fs/4, Fs/8
           - Concatenate -> BatchNorm -> ELU
           - DepthwiseConv2D (spatial) -> BatchNorm -> ELU -> AvgPool(1,8) -> Dropout
           - Conv2D(F2, (1,16)) -> BatchNorm -> ELU -> AvgPool(1,8) -> Dropout
        2. Attention Block:
           - Multi-Head Self-Attention with residual connection
        3. Classifier:
           - Flatten -> Dropout -> Dense(nb_classes) -> Softmax

    Parameters:
        nb_classes   : number of output classes
        Chans        : number of EEG channels
        Samples      : number of time samples
        F1           : number of filters per temporal branch (default 8)
        D            : depth multiplier for spatial conv (default 2)
        num_heads    : number of attention heads (default 2)
        dropout_rate : dropout fraction (default 0.5)
        Fs           : sampling frequency in Hz, used to set kernel sizes (default 250)
    """
    F2 = F1 * D  # feature dimension = 16

    # Temporal kernel sizes: half, quarter, eighth of sampling rate
    k1 = Fs // 2       # 125 for 250Hz
    k2 = Fs // 4       # 62  for 250Hz
    k3 = Fs // 8       # 31  for 250Hz

    input1 = Input(shape=(Chans, Samples, 1))

    # ---- Convolutional Block ----

    # Multi-scale temporal convolutions (3 parallel branches)
    branch1 = Conv2D(F1, (1, k1), padding='same', use_bias=False)(input1)
    branch2 = Conv2D(F1, (1, k2), padding='same', use_bias=False)(input1)
    branch3 = Conv2D(F1, (1, k3), padding='same', use_bias=False)(input1)

    # Concatenate along filter dimension -> (N, Chans, Samples, 3*F1)
    x = Concatenate(axis=-1)([branch1, branch2, branch3])
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Depthwise spatial convolution: (Chans, 1) -> collapses spatial dim
    x = DepthwiseConv2D((Chans, 1), depth_multiplier=1, use_bias=False,
                        depthwise_constraint=max_norm(1.))(x)
    # shape: (N, 1, Samples, 3*F1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    # shape: (N, 1, Samples/8, 3*F1)
    x = Dropout(dropout_rate)(x)

    # Final temporal convolution -> F2 filters
    x = Conv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    # shape: (N, 1, Samples/8, F2)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    # shape: (N, 1, Samples/64, F2)
    x = Dropout(dropout_rate)(x)

    # Reshape: (N, 1, Tsc, F2) -> (N, Tsc, F2) for attention
    x = Lambda(lambda t: t[:, 0, :, :])(x)

    # ---- Attention Block ----
    # Pre-norm Multi-Head Self-Attention with residual
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=F2 // num_heads
    )(x_norm, x_norm)
    x = Add()([x, attn_out])

    # ---- Classifier ----
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax')(x)

    return Model(inputs=input1, outputs=softmax)


def prepare_mcsanet_input(X, y):
    """
    Convert preprocessing output to MCSANet input format.

    preprocessing.py returns:
        X shape = (n_samples, n_times, n_channels)

    MCSANet expects:
        X shape = (n_samples, n_channels, n_times, 1)

    Labels converted from {1,2} -> {0,1}
    """
    X = np.transpose(X, (0, 2, 1))   # (N, T, C) -> (N, C, T)
    X = X[..., np.newaxis]            # (N, C, T) -> (N, C, T, 1)
    y = y.astype(int) - 1
    return X.astype(np.float32), y.astype(np.int64)


def run_mcsanet_cv(
    X,
    y,
    groups,
    config,
    n_splits=10,
    epochs=300,
    batch_size=16,
    learning_rate=1e-3,
):
    """
    Run MCSANet with cross-validation.
    """
    X, y = prepare_mcsanet_input(X, y)

    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running MCSANet =====")
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

        model = MCSANet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_samples,
            F1=8,
            D=2,
            num_heads=2,
            dropout_rate=0.5,
            Fs=250,
        )

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy'],
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
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

        print(f"MCSANet Fold {fold} accuracy: {acc:.4f}")
        print(f"MCSANet Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


if __name__ == "__main__":
    files = get_training_files("data/2b")

    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running MCSANet experiment with config:", config)

    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    mcsanet_accuracies, mcsanet_times = run_mcsanet_cv(
        X, y, groups, config=config, n_splits=10
    )

    print("\n===== MCSANet Final Results =====")
    print(f"Mean accuracy: {np.mean(mcsanet_accuracies):.4f}")
    print(f"Std accuracy : {np.std(mcsanet_accuracies):.4f}")
    print(f"Avg training time: {np.mean(mcsanet_times):.4f} s")
