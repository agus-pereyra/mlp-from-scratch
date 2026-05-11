import numpy as np

def cross_entropy(y_hat, y_onehot):
    y_hat_clipped = np.clip(y_hat, 1e-15, 1)
    return -np.mean(np.sum(y_onehot * np.log(y_hat_clipped), axis=1))

def accuracy(y_hat, y):
    y_pred = np.argmax(y_hat, axis=1)
    return np.mean(y_pred == y)

def confusion_matrix(y_hat, y, n_classes):
    y_pred = np.argmax(y_hat, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y, y_pred):
        cm[int(true), int(pred)] += 1
    return cm

def f1_macro(y_hat, y, n_classes):
    y_pred = np.argmax(y_hat, axis=1)
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y == k))
        fp = np.sum((y_pred == k) & (y != k))
        fn = np.sum((y_pred != k) & (y == k))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)