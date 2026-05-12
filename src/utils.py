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

def to_onehot(y, n_classes):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y.astype(int)] = 1
    return Y