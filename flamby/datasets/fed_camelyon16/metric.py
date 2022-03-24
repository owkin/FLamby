import numpy as np
from sklearn.metrics import roc_auc_score


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches \
    # have one class only
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan
