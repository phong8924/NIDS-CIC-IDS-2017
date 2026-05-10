import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_binary_data(data_path, load_test=True):
    X_train = joblib.load(os.path.join(data_path, "X_train.pkl"))
    y_train = joblib.load(os.path.join(data_path, "y_train.pkl"))
    class_weights = joblib.load(os.path.join(data_path, "class_weights.pkl"))
    result = (X_train, y_train, class_weights)
    if load_test:
        X_test = joblib.load(os.path.join(data_path, "X_test.pkl"))
        y_test = joblib.load(os.path.join(data_path, "y_test.pkl"))
        result = (X_train, y_train, X_test, y_test, class_weights)
    return result


def load_multiclass_data(data_path):
    X_train = joblib.load(os.path.join(data_path, "X_train_multi.pkl"))
    y_train = joblib.load(os.path.join(data_path, "y_train_multi.pkl"))
    X_test = joblib.load(os.path.join(data_path, "X_test_multi.pkl"))
    y_test = joblib.load(os.path.join(data_path, "y_test_multi.pkl"))
    class_weights = joblib.load(os.path.join(data_path, "class_weights_multi.pkl"))
    return X_train, y_train, X_test, y_test, class_weights


def reshape_for_cnn(X_train, X_test=None):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    result = (X_train_cnn,)
    if X_test is not None:
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        result = (X_train_cnn, X_test_cnn)
    return result


def train_val_split(X, y, val_split=0.1, shuffle=True):
    return train_test_split(
        X, y, test_size=val_split, random_state=42, shuffle=shuffle
    )


def load_parquet(path):
    return pd.read_parquet(path)


def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
