from itertools import product
from typing import Type
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from src.models import NN
from src.torch_models import TorchNN
from src.optimizers import Optimizer, GD
from src.metrics import f1_macro, accuracy

def batch_test(
        model: NN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_sizes: list[int],
        fit_params: dict = None,
        ) -> dict:
    """
    Entrena el modelo con cada batch_size de la lista y registra métricas.

    Parameters
    ----------
    model : NN
        Modelo instanciado.
    batch_sizes : list[int]
        Lista de tamaños de batch a comparar.
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, etc.).

    Returns
    -------
    results : dict
        Clave = batch_size. Valor = dict con:
        'train_loss', 'val_loss' (historiales), 'accuracy', 'f1_macro'.
    """
    n_classes = len(np.unique(y_train))
    default_fit = dict(epochs=200, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    m = deepcopy(model)
    
    results = {}
    bar = tqdm(batch_sizes, desc='Batch test', unit='batch', colour='blue', ncols=90)
    for bs in bar:
        bar.set_description(f'batch_size={bs}')
        m._param_init()
        m.optimizer.setup(m.weights, m.biases)
        history = m.fit(X_train, y_train, X_val, y_val, batch_size=bs, **default_fit)
        yhat_val = m.forward(X_val)
        results[bs] = {
            'train_loss' : history['train_loss'],
            'val_loss' : history['val_loss'],
            'accuracy' : accuracy(yhat_val, y_val),
            'f1_macro' : f1_macro(yhat_val, y_val, n_classes),
        }
        bar.set_postfix(val_loss=f'{history["val_loss"][-1]:.4f}')
    bar.colour = 'green'
    bar.refresh()
    return results

def grid_search(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict,
        optimizer_param_grid: dict = None,
        fit_params: dict = None,
        model_class: Type[NN | TorchNN] = NN,
        optimizer_class: Type[Optimizer] = GD,
        ) -> pd.DataFrame:
    """
    **Grid-Search** para estructura e hiperparámetros de cualquier modelo compatible (`NN` o `TorchNN`).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Datos de entrenamiento
    X_val, y_val : np.ndarray
        Datos de validación
    param_grid : dict
        Parámetros de arquitectura y entrenamiento a explorar (layers, dropout, batch_size, etc.)
    optimizer_param_grid : dict, optional
        Parámetros del optimizador a explorar (lr, weight_decay, etc.).
        Se cruzan con `param_grid` y se pasan como `optimizer_params` al modelo.
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, lr_schedule, etc.)
    model_class : type, default `NN`
        Clase del modelo a instanciar.
    optimizer_class : type, default `GD`
        Clase del optimizador que utilizará el modelo.

    Return
    ------
    results : pd.DataFrame
        DataFrame ordenado por val_loss ascendente (1 = mejor).
    """
    n_classes  = len(np.unique(y_train))
    input_size = X_train.shape[1]

    default_fit = dict(epochs=200, patience=20, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    opt_grid = optimizer_param_grid or {}
    fit_keys = list(param_grid.keys())
    opt_keys = list(opt_grid.keys())
    all_keys = fit_keys + opt_keys
    all_values = [param_grid[k] for k in fit_keys] + [opt_grid[k] for k in opt_keys]

    combos = list(product(*all_values))
    n_combos = len(combos)

    all_dims = ' x '.join(f"{k}({len(v)})" for k, v in {**param_grid, **opt_grid}.items())
    print(f"GRID SEARCH [{model_class.__name__} / {optimizer_class.__name__}] — {n_combos} models [{all_dims}]")

    best_val_loss = np.inf
    rows = []

    bar = tqdm(combos, total=n_combos, desc='Grid search', unit='model', colour='blue')

    for combo in bar:
        all_params = dict(zip(all_keys, combo))
        fit_combo = {k: all_params[k] for k in fit_keys}
        opt_combo = {k: all_params[k] for k in opt_keys}

        layers_original = fit_combo.pop('layers')
        layers_config = deepcopy(layers_original)
        layers_raw = deepcopy(layers_original)
        dropout = fit_combo.pop('dropout', None)

        arch_str = _fmt_layers(layers_raw)
        bar.set_description(arch_str)

        model = model_class(
            input_size, n_classes, layers_raw,
            optimizer=optimizer_class,
            optim_params=opt_combo,
            **({'dropout': dropout} if dropout is not None else {}),
        )
        history = model.fit(X_train, y_train, X_val, y_val, **{**default_fit, **fit_combo})

        val_f1 = f1_macro(model.forward(X_val), y_val, n_classes)
        val_loss = history['val_loss'][-1]
        epochs_trained = len(history['train_loss'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        bar.set_postfix(val_loss=f'{val_loss:.4f}', best=f'{best_val_loss:.4f}')

        row = {
            'layers' : arch_str,
            'layers_config' : layers_config,
            **({'dropout' : dropout} if dropout is not None else {}),
            **fit_combo,
            **opt_combo,
            'val_loss' : round(val_loss, 4),
            'val_f1_macro' : round(val_f1, 4),
            'epochs_trained' : epochs_trained,
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