import numpy as np
from sklearn.metrics import accuracy_score


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    y_pred = np.argmax(y_pred, axis=1)
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        return accuracy_score(y_true, y_pred)
    except ValueError:
        return np.nan
