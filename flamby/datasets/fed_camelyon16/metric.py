from sklearn.metrics import roc_auc_score


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    return roc_auc_score(y_true, y_pred)