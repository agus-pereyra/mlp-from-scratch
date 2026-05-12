from itertools import product
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from src.models import AdvancedNN
from src.metrics import f1_macro


def _fmt_layers(layers) -> str:
    """
    Formatear cómo se muestra la estructura de capas ocultas en la progress bar de grid_search
    
    [(64,'relu'),(32,'relu')] → '[64→32]'
    
    """
    sizes = [str(n) for n, _ in layers]
    return f"[{'→'.join(sizes)}]"


def grid_search(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict,
        fit_params: dict = None
        ) -> pd.DataFrame:
    """
    **Grid-Search** para estructura e hyperparámetros de `AdvancedNN`.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Datos de entrenamiento
    X_val, y_val : np.ndarray
        Datos de validación
    param_grid : dict
        Diccionario con parámetros.
        - *Key*: nombre del parámetro
        - *Value*: lista de valores a probar
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, lr_schedule, etc.)

    Returns
    -------
    results : pd.DataFrame
        DataFrame ordenado por f1-macro (de validation) descendente de los resultados. 
        Una columna por hiperparámetro ('layers' en formato compacto), más f1-macro,
        val_loss y epochs_trained. El índice representa el ranking (1 = mejor).
    """
    n_classes = len(np.unique(y_train))

    input_size = X_train.shape[1]

    default_fit = dict(epochs=200, patience=20, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    search_keys = list(param_grid.keys())
    search_values = [param_grid[k] for k in search_keys]

    combos = list(product(*search_values))
    n_combos = len(combos)

    grid_dims = ' x '.join(f"{k}({len(param_grid[k])})" for k in search_keys)
    print(f"GRID SEARCH — {n_combos} models [{grid_dims}]")

    best_f1 = -np.inf
    rows = []

    bar = tqdm(combos, total=n_combos, desc='Grid search', unit='model', colour='blue')

    for combo in bar:
        params = dict(zip(search_keys, combo))
        original = params.pop('layers') if 'layers' in params else [(64, 'relu')]
        layers_config = deepcopy(original)
        layers_raw = deepcopy(original)

        arch_str = _fmt_layers(layers_raw)
        bar.set_description(arch_str)

        model = AdvancedNN(input_size, n_classes, layers_raw)
        history = model.fit(X_train, y_train, X_val, y_val, **{**default_fit, **params})

        val_f1 = f1_macro(model.forward(X_val), y_val, n_classes)
        val_loss = history['val_loss'][-1]
        epochs_trained = len(history['train_loss'])

        if val_f1 > best_f1:
            best_f1 = val_f1

        bar.set_postfix(f1=f'{val_f1:.4f}', best_f1=f'{best_f1:.4f}')

        row = {'layers': arch_str,
               'layers_config': layers_config,
               **params,
               'val_f1_macro': round(val_f1, 4),
               'val_loss': round(val_loss,  4),
               'epochs_trained': epochs_trained}
        rows.append(row)

    bar.colour = 'green'
    bar.refresh()

    df = (pd.DataFrame(rows)
            .sort_values('val_f1_macro', ascending=False)
            .reset_index(drop=True))
    df.index += 1  # rank starting at 1
    return df
