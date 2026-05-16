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

def indexed_path(path):
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    i = 1
    while os.path.exists(f'{base}-{i}{ext}'):
        i += 1
    return f'{base}-{i}{ext}'

def to_onehot(y, n_classes):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y.astype(int)] = 1
    return Y