from itertools import product
from typing import Type
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from src.models import NN, AdvancedNN
from src.torch_models import TorchNN
from src.metrics import f1_macro

def grid_search(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict,
        fit_params: dict = None,
        model_class: Type[NN] = None,
        ) -> pd.DataFrame:
    """
    **Grid-Search** para estructura e hiperparámetros de cualquier modelo compatible
    (`NN`, `AdvancedNN` o `TorchNN`).

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
    model_class : type, optional
        Clase del modelo a instanciar. Por defecto `AdvancedNN`.

    Returns
    -------
    results : pd.DataFrame
        DataFrame ordenado por loss (de validation) descendente de los resultados.
        Una columna por hiperparámetro ('layers' en formato compacto), más f1-macro,
        val_loss y epochs_trained. El índice representa el ranking (1 = mejor).
    """
    if model_class is None:
        model_class = AdvancedNN

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
    print(f"GRID SEARCH [{model_class.__name__}] — {n_combos} models [{grid_dims}]")

    best_val_loss = np.inf
    rows = []

    bar = tqdm(combos, total=n_combos, desc='Grid search', unit='model', colour='blue')

    for combo in bar:
        params = dict(zip(search_keys, combo))
        layers_original = params.pop('layers')
        layers_config = deepcopy(layers_original)
        layers_raw = deepcopy(layers_original)
        dropout = params.pop('dropout', None)

        arch_str = _fmt_layers(layers_raw)
        bar.set_description(arch_str)

        # dropout solo lo admite TorchNN
        model = model_class(input_size, n_classes, layers_raw, **({'dropout': dropout} if dropout is not None else {}))
        history = model.fit(X_train, y_train, X_val, y_val, **{**default_fit, **params})

        val_f1 = f1_macro(model.forward(X_val), y_val, n_classes)
        val_loss = history['val_loss'][-1]
        epochs_trained = len(history['train_loss'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        bar.set_postfix(val_loss=f'{val_loss:.4f}', best=f'{best_val_loss:.4f}')

        row = {'layers':  arch_str,
               'layers_config': layers_config,
               'val_loss': round(val_loss, 4),
               **({'dropout': dropout} if dropout is not None else {}),
               **params,
               'val_f1_macro': round(val_f1, 4),
               'epochs_trained': epochs_trained
               }
        rows.append(row)

    bar.colour = 'green'
    bar.refresh()

    df = pd.DataFrame(rows).sort_values('val_loss', ascending=True).reset_index(drop=True)
    df.index += 1
    
    return df

def _fmt_layers(layers) -> str:
    """
    Formatear cómo se muestra la estructura de capas ocultas en la progress bar de grid_search

    [(64,'relu'),(32,'relu')] → '[64→32]'

    """
    sizes = [str(n) for n, _ in layers]
    return f"[{'→'.join(sizes)}]"