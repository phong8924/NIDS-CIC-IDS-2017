from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def build_dnn_binary(input_dim, learning_rate=0.001):
    model = Sequential(name="DNN_Binary")
    model.add(Dense(512, input_dim=input_dim, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dnn_multiclass(input_dim, n_classes, use_onehot=True, learning_rate=0.001):
    loss = "categorical_crossentropy" if use_onehot else "sparse_categorical_crossentropy"
    model = Sequential(name="DNN_Multiclass")
    model.add(Dense(512, input_shape=(input_dim,), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model
