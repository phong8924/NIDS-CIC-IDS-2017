import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import EARLY_STOPPING_PATIENCE, LR_PATIENCE, LR_FACTOR, MIN_LR


def build_callbacks(
    checkpoint_path: str,
    early_stop_patience: int = EARLY_STOPPING_PATIENCE,
    lr_patience: int = LR_PATIENCE,
    lr_factor: float = LR_FACTOR,
    min_lr: float = MIN_LR,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]
