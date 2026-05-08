import time
import tempfile
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Permute, Conv2D, Conv1D, DepthwiseConv2D,
    BatchNormalization, Activation, AveragePooling2D, Dropout,
    Dense, Lambda, Add, Concatenate, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, cohen_kappa_score

from preprocessing import (
    PreprocessingConfig,
    get_training_files,
    preprocess_subject_windows,
)


# ---------------------------------------------------------------------------
# Exact copy of mha_block from Altaheri/EEG-ATCNet attention_models.py
# ---------------------------------------------------------------------------
def mha_block(input_feature, key_dim=8, num_heads=2, dropout=0.5):
    """Multi-head self-attention block (vanilla MHA).
    Copied from https://github.com/Altaheri/EEG-ATCNet/blob/main/attention_models.py
    """
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(0.3)(x)
    return Add()([input_feature, x])


# ---------------------------------------------------------------------------
# Exact copy of Conv_block_ from Altaheri/EEG-ATCNet models.py
# ---------------------------------------------------------------------------
def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    """Convolutional block with regularization.
    Copied from https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    """
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    block2 = DepthwiseConv2D((1, in_chans),
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                             use_bias=False)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = Conv2D(F2, (16, 1), data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


# ---------------------------------------------------------------------------
# Exact copy of TCN_block_ from Altaheri/EEG-ATCNet models.py
# ---------------------------------------------------------------------------
def TCN_block_(input_layer, input_dimension, depth, kernel_size, filters, dropout,
               weightDecay=0.009, maxNorm=0.6, activation='relu'):
    """Temporal convolutional block with regularization.
    Copied from https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    """
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)

    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1,
                      kernel_regularizer=L2(weightDecay),
                      kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                      padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                       activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                       activation='linear',
                       kernel_regularizer=L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        added = Add()([block, out])
        out = Activation(activation)(added)

    return out


# ---------------------------------------------------------------------------
# Exact copy of ATCNet_ from Altaheri/EEG-ATCNet models.py
# ---------------------------------------------------------------------------
def ATCNet(n_classes, in_chans=22, in_samples=1125, n_windows=5,
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):
    """ATCNet model — exact copy of ATCNet_ from Altaheri et al. 2022.
    https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    Input shape: (None, 1, in_chans, in_samples)
    """
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    dense_weightDecay = 0.5
    conv_weightDecay  = 0.009
    conv_maxNorm      = 0.6
    F2 = eegn_F1 * eegn_D

    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    sw_concat = []
    for i in range(n_windows):
        st  = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        block2 = mha_block(block2)

        block3 = TCN_block_(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif fuse == 'concat':
            sw_concat = block3 if i == 0 else Concatenate()([sw_concat, block3])

    if fuse == 'average':
        out = tf.keras.layers.Average()(sw_concat) if len(sw_concat) > 1 else sw_concat[0]
    elif fuse == 'concat':
        out = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(sw_concat)

    out = Activation('softmax', name='softmax')(out)
    return Model(inputs=input_1, outputs=out)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def prepare_input(X, y):
    """Convert (N, T, C) → (N, 1, C, T) and map labels {1..K} → {0..K-1}."""
    X = np.transpose(X, (0, 2, 1))
    X = X[:, np.newaxis, :, :]
    y = y.astype(int) - 1
    return X.astype(np.float32), y.astype(np.int64)


def standardize(X_train, X_test):
    """Per-channel StandardScaler — exact copy of standardize_data from preprocess.py."""
    from sklearn.preprocessing import StandardScaler
    n_channels = X_train.shape[2]
    for j in range(n_channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :]  = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def run_atcnet_holdout(X_train, y_train, X_test, y_test,
                       epochs=500, batch_size=64, learning_rate=1e-3,
                       augment=True):
    """
    Train on X_train, evaluate on X_test.

    Matches official ATCNet repo (Altaheri et al. 2022/2023):
    - No bandpass filter — raw signal for BCI2a
    - No downsampling — 250 Hz
    - Full window (1125 samples for 2a, whatever is passed in)
    - Per-channel StandardScaler normalisation
    - 500 epochs, no early stopping
    - ModelCheckpoint restores best val_loss weights
    - ReduceLROnPlateau: factor=0.90, patience=20, min_lr=1e-4
    """
    X_train, y_train = prepare_input(X_train, y_train)
    X_test,  y_test  = prepare_input(X_test,  y_test)
    X_train, X_test  = standardize(X_train, X_test)

    n_classes  = len(np.unique(y_train))
    n_channels = X_train.shape[2]
    n_samples  = X_train.shape[3]

    model = ATCNet(n_classes=n_classes, in_chans=n_channels, in_samples=n_samples,
                   n_windows=5, eegn_F1=16, eegn_D=2, eegn_kernelSize=64,
                   eegn_poolSize=7, eegn_dropout=0.3, tcn_depth=2,
                   tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                   tcn_activation='elu', fuse='average')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90,
                                  patience=20, min_lr=1e-4, verbose=0)
    tmpfile = tempfile.mktemp(suffix='.weights.h5')
    checkpoint = ModelCheckpoint(tmpfile, monitor='val_loss',
                                 save_best_only=True, save_weights_only=True, verbose=0)

    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[reduce_lr, checkpoint], verbose=0)
    end = time.time()

    model.load_weights(tmpfile)
    os.remove(tmpfile)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_test, y_pred)
    kappa  = cohen_kappa_score(y_test, y_pred)
    print(f"ATCNet accuracy: {acc:.4f}  kappa: {kappa:.2f}  time: {end-start:.1f}s")
    return acc, kappa, end - start


if __name__ == "__main__":
    files  = get_training_files("data/2b")
    config = PreprocessingConfig(A=1, B=2, C=1, D=2)

    print("Running ATCNet experiment with config:", config)
    X, y, groups = preprocess_subject_windows(files[0], config)

    print("\nDataset loaded")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    print("\nUse run_all.py to run experiments with the T-E holdout protocol.")
