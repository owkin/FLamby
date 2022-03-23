from sklearn.metrics import roc_auc_score


def metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)