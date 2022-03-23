from sklearn.metrics import roc_auc_score
import numpy as np


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batch some batches have one class only
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return np.nan