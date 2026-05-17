import os
import numpy as np

LABELS = [
    'お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を',
    'あ', 'い', 'う', 'え', 'か', 'く', 'け', 'こ', 'さ', 'し',
    'せ', 'そ', 'た', 'ち', 'に', 'ぬ', 'ね', 'の', 'ひ', 'ふ',
    'へ', 'ほ', 'み', 'む', 'め', 'も', 'ゆ', 'よ', 'ら', 'り',
    'る', 'ろ', 'わ', 'ゐ', 'ゑ', 'よ', 'ん', '゛', '゜'
]

LABELS_IDX_Y = [f'{c}→{idx}' for idx, c in enumerate(LABELS)]
LABELS_IDX_X = [f'{idx}\n↓\n{c}' for idx, c in enumerate(LABELS)]

def indexed_path(path: str) -> str:
    """
    Devuelve un path que no sobreescriba archivos existentes, agregando un índice numérico.

    Parameters
    ----------
    path : str
        Path deseado para el archivo.

    Returns
    -------
    path : str
        El mismo path si no existe, o 'base-1.ext', 'base-2.ext', etc. en caso contrario.
    """
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    i = 1
    while os.path.exists(f'{base}-{i}{ext}'):
        i += 1
    return f'{base}-{i}{ext}'

def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convierte un vector de labels enteros a codificación one-hot.

    Parameters
    ----------
    y : np.ndarray of shape (N,)
        Vector de labels enteros en el rango [0, n_classes).
    n_classes : int
        Número total de clases.

    Returns
    -------
    Y : np.ndarray of shape (N, n_classes)
        Matriz one-hot donde `Y[i, y[i]] = 1` y el resto es 0.
    """
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y.astype(int)] = 1
    return Y