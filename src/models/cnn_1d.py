from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Activation,
)
from tensorflow.keras.optimizers import Adam


def _cnn_2block_base(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    return model


def build_cnn_1d_binary(input_shape, learning_rate=0.001):
    model = _cnn_2block_base(input_shape)
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_1d_multiclass(input_shape, n_classes, learning_rate=0.001):
    model = _cnn_2block_base(input_shape)
    model.add(BatchNormalization())
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_3block(input_shape, n_classes, learning_rate=0.0001):
    model = Sequential(name="CNN_3Block")
    model.add(Conv1D(filters=128, kernel_size=7, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
