from itertools import product
from typing import Type
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from src.models import NN
from src.torch_models import TorchNN
from src.optimizers import Optimizer, GD
from src.metrics import f1_macro, accuracy, cross_entropy
from src.utils import to_onehot

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
        Clave = batch_size. 
        Valor = dict con keys 'train_loss', 'val_loss' (historiales), 'accuracy', 'f1_macro'.
    """
    n_classes = len(np.unique(y_train))
    default_fit = dict(epochs=400, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    m = deepcopy(model)
    m.optimizer.lr = 0.1
    
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

def optimizer_test(
        model: NN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimizers: list[Type[Optimizer]],
        parameters: list[dict],
        fit_params: dict = None,
        ) -> dict:
    """
    Entrena el modelo con distintos optimizadores y sus combinaciones de hiperparámetros.

    Parameters
    ----------
    model : NN
        Modelo base.
    optimizers : list[Type[Optimizer]]
        Lista de clases de optimizador a comparar.
    parameters : list[dict]
        Un dict por optimizador. Si un valor es una lista, se prueba una variante por cada elemento.
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, etc.).
    Returns
    -------
    results : dict
        Clave = etiqueta descriptiva del experimento (e.g. 'Adam(lr=0.01)').
        Valor = dict con 'train_loss', 'val_loss', 'accuracy', 'f1_macro'.
    """
    n_classes = len(np.unique(y_train))
    default_fit = dict(epochs=400, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    runs = []
    for optim_class, params in zip(optimizers, parameters):
        list_keys = [k for k, v in params.items() if isinstance(v, list)]
        scalar_keys = [k for k, v in params.items() if not isinstance(v, list)]

        if not list_keys:
            runs.append((optim_class, dict(params)))
        else:
            list_values = [params[k] for k in list_keys]
            for combo in product(*list_values):
                p = {k: params[k] for k in scalar_keys}
                p.update(dict(zip(list_keys, combo)))
                runs.append((optim_class, p))

    m = deepcopy(model)
    results = {}
    bar = tqdm(runs, desc='Optimizer test', unit='model', colour='blue', ncols=110)
    for optim_class, params in bar:
        label = f"{optim_class.__name__}({', '.join(f'{k}={v}' for k, v in params.items())})"
        bar.set_description(label)

        m._param_init()
        m.optimizer = optim_class(**params)
        m.optimizer.setup(m.weights, m.biases)

        history = m.fit(X_train, y_train, X_val, y_val, **default_fit)
        yhat_val = m.forward(X_val)
        results[label] = {
            'optimizer' : optim_class.__name__,
            'params' : params,
            'train_loss' : history['train_loss'],
            'val_loss' : history['val_loss'],
            'accuracy' : accuracy(yhat_val, y_val),
            'f1_macro' : f1_macro(yhat_val, y_val, n_classes),
        }
        bar.set_postfix(val_loss=f'{history["val_loss"][-1]:.4f}')

    bar.colour = 'green'
    bar.refresh()
    return results

def lr_scheduling_test(
        model: NN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lr_mins: list[float],
        gammas: list[float],
        fit_params: dict = None,
        ) -> tuple[dict, dict]:
    """
    Prueba scheduling lineal y exponencial para distintos valores de lr_min y gamma.

    Parameters
    ----------
    model : NN
        Modelo base (se reinicializa por cada variante).
    lr_mins : list[float]
        Valores de lr_min a probar (usados en ambos schedulers).
    gammas : list[float]
        Valores de gamma a probar (solo exponential; ignorado en linear).
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, etc.).

    Returns
    -------
    constant_results, linear_results, exponential_results : tuple[dict, dict, dict]
        Cada dict tiene la misma estructura que batch_test/optimizer_test:
        clave = etiqueta descriptiva, valor = dict con train_loss, val_loss, accuracy, f1_macro.
    """
    n_classes = len(np.unique(y_train))
    default_fit = dict(epochs=400, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    m = deepcopy(model)
    constant_results = {}
    linear_results = {}
    exponential_results = {}

    linear_runs = [(lr_min,) for lr_min in lr_mins]
    exponential_runs = [(gamma,) for gamma in gammas]
    all_runs = [('constant', 'constant lr', {})] + \
               [('linear', f'lr_min={v}', dict(lr_schedule='linear', lr_min=v)) for (v,) in linear_runs] + \
               [('exp', f'gamma={v}', dict(lr_schedule='exponential', gamma=v)) for (v,) in exponential_runs]

    bar = tqdm(all_runs, desc='LR scheduling test', unit='model', colour='blue', ncols=100)
    for kind, label, schedule_params in bar:
        bar.set_description(f'{kind}({label})')

        m._param_init()
        m.optimizer.setup(m.weights, m.biases)

        history = m.fit(X_train, y_train, X_val, y_val, **{**default_fit, **schedule_params})
        yhat_val = m.forward(X_val)
        entry = {
            'train_loss' : history['train_loss'],
            'val_loss' : history['val_loss'],
            'lr' : history.get('lr', []),
            'accuracy' : accuracy(yhat_val, y_val),
            'f1_macro' : f1_macro(yhat_val, y_val, n_classes),
        }
        bar.set_postfix(val_loss=f'{history["val_loss"][-1]:.4f}')

        if kind == 'constant':
            constant_results[label] = entry
        elif kind == 'linear':
            linear_results[label] = entry
        else:
            exponential_results[label] = entry

    bar.colour = 'green'
    bar.refresh()
    return constant_results, linear_results, exponential_results

def weight_decay_test(
        model: NN,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimizer_info: dict[Type[Optimizer], dict],
        weight_decays: list[float],
        fit_params: dict = None,
        ) -> dict:
    """
    Entrena el modelo con distintos valores de weight_decay para un optimizador dado.

    Parameters
    ----------
    model : NN
        Modelo base (se reinicializa por cada variante).
    optimizer_info : dict[Type[Optimizer], dict]
        {OptimizerClass: {param: value, ...}} con los parámetros base de cada optimizador.
        Puede contener múltiples optimizadores; se prueban todos los weight_decay para cada uno.
    weight_decays : list[float]
        Valores de weight_decay a probar.
    fit_params : dict, optional
        Parámetros fijos del entrenamiento (epochs, patience, etc.).

    Returns
    -------
    results : dict
        Clave = 'OptimizerName(wd=value)'. Valor = dict con train_loss, val_loss, accuracy, f1_macro.
    """
    n_classes = len(np.unique(y_train))
    default_fit = dict(epochs=400, verbose=False)
    if fit_params:
        default_fit.update(fit_params)

    runs = [(opt_cls, base_params, wd)
            for opt_cls, base_params in optimizer_info.items()
            for wd in weight_decays]

    m = deepcopy(model)
    results = {}
    bar = tqdm(runs, desc='Weight decay test', unit='model', colour='blue', ncols=110)
    for opt_cls, base_params, wd in bar:
        label = f'{opt_cls.__name__}(wd={wd})'
        bar.set_description(label)

        m._param_init()
        m.optimizer = opt_cls(**base_params, weight_decay=wd)
        m.optimizer.setup(m.weights, m.biases)

        history = m.fit(X_train, y_train, X_val, y_val, **default_fit)
        yhat_val = m.forward(X_val)
        results[label] = {
            'weight_decay' : wd,
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

def perturb(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Agrega ruido gaussiano a los datos y recorta los valores al rango [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Datos de entrada normalizados en [0, 1].
    sigma : float
        Desviación estándar del ruido gaussiano a aplicar.

    Returns
    -------
    X_noisy : np.ndarray
        Datos perturbados, del mismo shape que `X`, con valores en [0, 1].
    """
    noise = np.random.normal(0, sigma, X.shape)
    return np.clip(X + noise, 0, 1)

def initialization_variance(model : NN | TorchNN, fit_params: dict,
                            X_train, y_train, X_val, y_val,
                            n_runs: int = 10, n_classes: int = 49) -> dict:
    """
    Re-entrena el modelo `n_runs` veces con distintas inicializaciones aleatorias
    de parámetros, manteniendo los mismos hiperparámetros, y registra las métricas
    finales sobre validación en cada corrida.

    Parameters
    ----------
    model : NN | TorchNN
        Modelo base cuya arquitectura e hiperparámetros se reutilizan.
    fit_params : dict
        Parámetros de entrenamiento a mantener constantes (epochs, batch_size, etc.).
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_val, y_val : np.ndarray
        Datos de validación sobre los que se evalúa cada corrida.
    n_runs : int, default `10`
        Número de re-inicializaciones y re-entrenamientos a realizar.
    n_classes : int, default `49`
        Número de clases del problema.

    Returns
    -------
    records : dict
        Diccionario con listas de métricas por corrida.
        Keys: 'accuracy', 'cross_entropy', 'f1_macro'.
    """
    records = {'accuracy': [], 'cross_entropy': [], 'f1_macro': []}
    y_val_oh = to_onehot(y_val, n_classes)
    for i in range(1, n_runs + 1):
        m = deepcopy(model)
        m._param_init()
        m.fit(X_train, y_train, X_val, y_val, **{**fit_params, 'verbose': False})
        yhat = m.forward(X_val)
        records['accuracy'].append(accuracy(yhat, y_val))
        records['cross_entropy'].append(cross_entropy(yhat, y_val_oh))
        records['f1_macro'].append(f1_macro(yhat, y_val, n_classes))
    return records

def robustness_test(
        models: list[NN | TorchNN],
        names: list[str],
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: list[float],
        n_classes: int,
        ) -> pd.DataFrame:
    """
    Evalúa la robustez de una lista de modelos ante distintos niveles de ruido gaussiano.

    Para cada nivel de ruido `sigma`, perturba los datos con `perturb` y calcula
    accuracy, F1-macro y cross-entropy de cada modelo sobre los datos perturbados.

    Parameters
    ----------
    models : list[NN | TorchNN]
        Lista de modelos entrenados a evaluar.
    names : list[str]
        Etiquetas identificadoras de cada modelo (mismo orden que `models`).
    X : np.ndarray
        Datos de entrada normalizados en [0, 1] sobre los que se aplica el ruido.
    y : np.ndarray
        Labels verdaderos correspondientes a `X`.
    noise_levels : list[float]
        Valores de sigma a probar. `0` evalúa los datos sin perturbación.
    n_classes : int
        Número de clases del problema.

    Returns
    -------
    results : pd.DataFrame
        DataFrame con columnas 'model', 'sigma', 'accuracy', 'f1_macro', 'cross_entropy'.
        Una fila por cada combinación (modelo, sigma).
    """
    y_oh = to_onehot(y, n_classes)
    rows = []
    for sigma in noise_levels:
        X_noisy = perturb(X, sigma) if sigma > 0 else X
        for model, name in zip(models, names):
            yhat = model.forward(X_noisy)
            rows.append({
                'model' : name,
                'sigma' : sigma,
                'accuracy' : accuracy(yhat, y),
                'f1_macro' : f1_macro(yhat, y, n_classes),
                'cross_entropy': cross_entropy(yhat, y_oh),
            })
    return pd.DataFrame(rows)

def _fmt_layers(layers) -> str:
    """
    Formatear cómo se muestra la estructura de capas ocultas en la progress bar de grid_search

    Ejemplo: [(64,'relu'),(32,'relu')] → '[64→32]'
    """
    sizes = [str(n) for n, _ in layers]
    return f"[{'→'.join(sizes)}]"