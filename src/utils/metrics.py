import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def measure_inference_time(model, X_data, num_samples=1000, verbose=False):
    n = min(num_samples, len(X_data))
    indices = np.random.choice(len(X_data), n, replace=False)
    X_sample = X_data[indices]

    _ = model.predict(X_sample[:10], verbose=0)

    start = time.time()
    _ = model.predict(X_sample, verbose=0)
    elapsed = time.time() - start

    us_per_sample = (elapsed / n) * 1_000_000
    if verbose:
        print(f"   - {us_per_sample:.2f} µs/mẫu ({n} mẫu)")
    return us_per_sample


def evaluate_classification(y_true, y_pred, target_names=None):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return cm, report
