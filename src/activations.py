import numpy as np

def relu(a: np.ndarray) -> np.ndarray:
    """
    Función de activación ReLU: `max(0, a)`.

    Parameters
    ----------
    a : np.ndarray
        Pre-activaciones de la capa.

    Returns
    -------
    np.ndarray
        Activaciones, del mismo shape que `a`.
    """
    return np.maximum(0, a)

def drelu(a: np.ndarray) -> np.ndarray:
    """
    Derivada de ReLU: 1 si a > 0, 0 en caso contrario.

    Parameters
    ----------
    a : np.ndarray
        Pre-activaciones de la capa.

    Returns
    -------
    np.ndarray
        Gradiente elemento a elemento, del mismo shape que `a`.
    """
    return (a > 0).astype(float)

def softmax(a: np.ndarray) -> np.ndarray:
    """
    Función de activación Softmax, numéricamente estable por desplazamiento del máximo.

    Parameters
    ----------
    a : np.ndarray of shape (N, C)
        Pre-activaciones de la capa de salida.

    Returns
    -------
    np.ndarray of shape (N, C)
        Distribución de probabilidad sobre las C clases para cada muestra.
    """
    a_shifted = a - np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a_shifted)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def linear(a: np.ndarray) -> np.ndarray:
    """
    Activación lineal (identidad): `f(a) = a`.

    Parameters
    ----------
    a : np.ndarray
        Pre-activaciones de la capa.

    Returns
    -------
    np.ndarray
        El mismo array `a` sin modificación.
    """
    return a

def dlinear(a: np.ndarray) -> np.ndarray:
    """
    Derivada de la activación lineal: constante 1 en todo el dominio.

    Parameters
    ----------
    a : np.ndarray
        Pre-activaciones de la capa.

    Returns
    -------
    np.ndarray
        Array de unos del mismo shape que `a`.
    """
    return np.ones_like(a)

ACTIVATIONS = {
    'relu':    (relu, drelu),
    'softmax': (softmax, None),
    'linear':  (linear, dlinear),
}