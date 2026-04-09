import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Permute, Conv2D, Conv1D, DepthwiseConv2D,
    BatchNormalization, Activation, AveragePooling2D, Dropout,
    Dense, Lambda, Add, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.regularizers import L2
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


def conv_block_atc(input_layer, F1=16, kernLength=32, poolSize=7, D=2, in_chans=3,
                   weightDecay=0.009, maxNorm=0.6, dropout=0.3):
    """
    Convolution block from ATCNet, adapted for 2b.
    Input after permute: (None, Samples, Chans, 1)
    """
    F2 = F1 * D

    x = Conv2D(
        F1, (kernLength, 1), padding='same',
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
        use_bias=False
    )(input_layer)
    x = BatchNormalization(axis=-1)(x)

    x = DepthwiseConv2D(
        (1, in_chans),
        depth_multiplier=D,
        depthwise_regularizer=L2(weightDecay),
        depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
        use_bias=False
    )(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropout)(x)

    x = Conv2D(
        F2, (16, 1), padding='same',
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
        use_bias=False
    )(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((poolSize, 1))(x)
    x = Dropout(dropout)(x)

    return x


def mha_block(x, num_heads=2, key_dim=8, dropout=0.1):
    """
    Simple self-attention block.
    Input shape: (None, seq_len, features)
    """
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    return x


def tcn_block(input_layer, input_dimension, depth=2, kernel_size=4, filters=32,
              weightDecay=0.009, maxNorm=0.6, dropout=0.3, activation='elu'):
    x = Conv1D(
        filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
        padding='causal', kernel_initializer='he_uniform'
    )(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)

    x = Conv1D(
        filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
        padding='causal', kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)

    if input_dimension != filters:
        shortcut = Conv1D(
            filters, kernel_size=1, padding='same',
            kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1])
        )(input_layer)
        out = Add()([x, shortcut])
    else:
        out = Add()([x, input_layer])

    out = Activation(activation)(out)

    for i in range(depth - 1):
        x = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
            kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
            padding='causal', kernel_initializer='he_uniform'
        )(out)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        x = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
            kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
            padding='causal', kernel_initializer='he_uniform'
        )(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        out = Add()([x, out])
        out = Activation(activation)(out)

    return out


def ATCNet(n_classes, in_chans=3, in_samples=1000, n_windows=5,
           eegn_F1=16, eegn_D=2, eegn_kernelSize=32, eegn_poolSize=7, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):
    """
    Simplified ATCNet adapted for BCI IV 2b.
    Input shape: (None, 1, in_chans, in_samples)
    """
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)   # -> (None, Samples, Chans, 1)

    dense_weightDecay = 0.5
    F2 = eegn_F1 * eegn_D

    block1 = conv_block_atc(
        input_layer=input_2,
        F1=eegn_F1,
        D=eegn_D,
        kernLength=eegn_kernelSize,
        poolSize=eegn_poolSize,
        in_chans=in_chans,
        dropout=eegn_dropout
    )

    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)  # -> (None, seq_len, F2)

    sw_outputs = []
    seq_len = block1.shape[1]

    for i in range(n_windows):
        st = i
        end = seq_len - n_windows + i + 1
        x = block1[:, st:end, :]

        x = mha_block(x, num_heads=2, key_dim=8, dropout=0.1)

        x = tcn_block(
            input_layer=x,
            input_dimension=F2,
            depth=tcn_depth,
            kernel_size=tcn_kernelSize,
            filters=tcn_filters,
            dropout=tcn_dropout,
            activation=tcn_activation
        )

        x = Lambda(lambda z: z[:, -1, :])(x)

        if fuse == 'average':
            x = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(x)
            sw_outputs.append(x)
        else:
            sw_outputs.append(x)

    if fuse == 'average':
        if len(sw_outputs) > 1:
            out = tf.keras.layers.Average()(sw_outputs)
        else:
            out = sw_outputs[0]
    else:
        out = tf.keras.layers.Concatenate()(sw_outputs)
        out = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(out)

    out = Activation('softmax')(out)

    return Model(inputs=input_1, outputs=out)


def run_atcnet_cv(X, y, groups, config, n_splits=10, epochs=100, batch_size=16, learning_rate=1e-3):
    X, y = prepare_input(X, y)
    splits = make_cv_splits(X, y, config=config, groups=groups, n_splits=n_splits)

    accuracies = []
    times = []

    print("\n===== Running ATCNet =====")
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

        model = ATCNet(
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=n_samples,
            n_windows=5,
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu',
            fuse='average'
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=15,
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

        print(f"ATCNet Fold {fold} accuracy: {acc:.4f}")
        print(f"ATCNet Fold {fold} training time: {train_time:.4f} s\n")

    return accuracies, times


if __name__ == "__main__":
    files = get_training_files("data")
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running ATCNet experiment with config:", config)

    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("groups is None:", groups is None)

    accs, ts = run_atcnet_cv(
        X, y, groups,
        config=config,
        n_splits=10,
        epochs=100,
        batch_size=16,
        learning_rate=1e-3,
    )

    print("\n===== ATCNet Final Results =====")
    print(f"Mean accuracy: {np.mean(accs):.4f}")
    print(f"Std accuracy : {np.std(accs):.4f}")
    print(f"Avg training time: {np.mean(ts):.4f} s")