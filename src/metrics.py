import numpy as np

def cross_entropy(y_hat: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Entropía Cruzada promedio sobre el batch.

    Parameters
    ----------
    y_hat : np.ndarray of shape (N, C)
        Probabilidades predichas por el modelo (output de softmax).
    y_onehot : np.ndarray of shape (N, C)
        Labels verdaderos en codificación one-hot.

    Returns
    -------
    loss : float
        Valor escalar de la entropía cruzada media sobre las N muestras.
    """
    y_hat_clipped = np.clip(y_hat, 1e-15, 1) # evitar log(0)
    return -np.mean(np.sum(y_onehot * np.log(y_hat_clipped), axis=1))

def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Fracción de predicciones correctas sobre el total de muestras.

    Parameters
    ----------
    y_hat : np.ndarray of shape (N, C)
        Probabilidades predichas por el modelo (output de softmax).
    y : np.ndarray of shape (N,)
        Labels verdaderos como enteros.

    Returns
    -------
    acc : float
        Accuracy en el rango [0, 1].
    """
    y_pred = np.argmax(y_hat, axis=1)
    return np.mean(y_pred == y)

def confusion_matrix(y_hat: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Matriz de confusión de forma (n_classes, n_classes).

    Parameters
    ----------
    y_hat : np.ndarray of shape (N, C)
        Probabilidades predichas por el modelo (output de softmax).
    y : np.ndarray of shape (N,)
        Labels verdaderos como enteros.
    n_classes : int
        Número total de clases.

    Returns
    -------
    cm : np.ndarray of shape (n_classes, n_classes)
        `cm[i, j]` contiene el número de muestras de la clase `i` clasificadas como `j`.
    """
    y_pred = np.argmax(y_hat, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y, y_pred):
        cm[int(true), int(pred)] += 1
    return cm

def f1_macro(y_hat: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """
    F1-Score Macro: promedio no ponderado de F1-Score por clase.

    Calcula precision, recall y F1 para cada clase individualmente
    y devuelve su media aritmética. Las clases sin muestras ni predicciones
    contribuyen con F1 = 0.

    Parameters
    ----------
    y_hat : np.ndarray of shape (N, C)
        Probabilidades predichas por el modelo (output de softmax).
    y : np.ndarray of shape (N,)
        Labels verdaderos como enteros.
    n_classes : int
        Número total de clases.

    Returns
    -------
    f1 : float
        F1-macro en el rango [0, 1].
    """
    y_pred = np.argmax(y_hat, axis=1)
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y == k))
        fp = np.sum((y_pred == k) & (y != k))
        fn = np.sum((y_pred != k) & (y == k))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)