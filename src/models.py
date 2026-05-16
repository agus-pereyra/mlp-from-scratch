import time
import numpy as np
from tqdm import tqdm
from typing import Literal, Type
from copy import deepcopy

from src.activations import ACTIVATIONS
from src.optimizers import Optimizer, SGD
from src.utils import to_onehot
from src.metrics import cross_entropy

class NN:
    '''
    ***Neural Network*** / ***Multi-layer Perceptron*** para clasificación multiclase
    con Softmax como capa de salida Entropía Cruzada como loss del modelo.
    '''
    def __init__(self, input_size: int, output_size: int, layers: list[tuple],
                 optimizer: Type[Optimizer]=SGD, optim_params: dict = None):
        """
        Parameters
        ----------
        input_size : int
            Tamaño del vector input `x`
        output_size : int
            Tamaño del vector output `y`
        layers : list[tuple]
            Lista de tuplas `(n_neurons, activation_name)` representando cada capa oculta
        optimizer : type, default `SGD`
            Clase del optimizador a usar
        optim_params : dict, optional
            Parámetros del optimizador (e.g. `{'lr': 1e-3, 'weight_decay': 1e-4}`)
        """
        self.input_size = input_size
        self.layers = deepcopy(layers)
        self.layers.append((output_size, 'softmax'))
        self.weights = []
        self.biases = []

        if optim_params:
            self.optimizer = optimizer(**optim_params)
        else:
            self.optimizer = optimizer(**{})
        self._param_init()

    def _param_init(self):
        '''
        He initialization (sigma^2 = 2/n_in)
        '''
        self.weights = []
        self.biases = []
        in_size = self.input_size
        for (out_size, _) in self.layers:
            self.weights.append(np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size))
            self.biases.append(np.zeros((1, out_size)))
            in_size = out_size

    def forward(self, X: np.ndarray):
        '''
        ***Forward Pass***

        Cálculo de pre-activaciones (a) y activaciones (z) de la red para los datos X
        y los pesos actuales del modelo.

        Note
        ----
        Algoritmo basado en el Algoritmo 8.1 de "Deep Learning: Foundations and Concepts -- C.M & H Bishop"
        adaptado de forma matricial.
        '''
        self.z_cache = [X]  # z[0] = x
        self.a_cache = []

        z = X
        for i, (_, act_name) in enumerate(self.layers):
            a = z @ self.weights[i] + self.biases[i]
            phi, _ = ACTIVATIONS[act_name]
            z = phi(a)
            self.a_cache.append(a)
            self.z_cache.append(z)

        return z

    def backward(self, y_onehot: np.ndarray):
        '''
        ***Backward Pass***

        Computa y almacena los gradientes de la pérdida respecto a los pesos y biases.
        No actualiza los parámetros — eso es responsabilidad del optimizador.

        Note
        ----
        Algoritmo basado en el Algoritmo 8.1 de "Deep Learning: Foundations and Concepts -- C.M & H Bishop"
        adaptado de forma matricial.
        '''
        m = y_onehot.shape[0]
        L = len(self.layers)
        deltas = [None] * L

        deltas[L - 1] = self.z_cache[L] - y_onehot  # output layer

        for l in range(L - 2, -1, -1):               # hidden layers
            _, act_deriv = ACTIVATIONS[self.layers[l][1]]
            deltas[l] = (deltas[l + 1] @ self.weights[l + 1].T) * act_deriv(self.a_cache[l])

        self.dW = [self.z_cache[l].T @ deltas[l] / m for l in range(L)] # /m (batch_size)
        self.db = [np.sum(deltas[l], axis=0, keepdims=True) / m for l in range(L)]

    def update(self):
        self.optimizer.step(self.weights, self.biases, self.dW, self.db)

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            epochs: int = 500,
            batch_size: int = None,
            lr_schedule: Literal['linear', 'exponential'] = None,
            lr_min: float = 1e-5,
            gamma: float = 0.99,
            patience: int = None,
            verbose: bool = True,
            ):
        '''
        ***Entrenamiento*** de la red.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Datos de entrenamiento
        X_val, y_val : np.ndarray, optional
            Datos de validación. Necesarios para early stopping e historial de val_loss.
        epochs : int, default `500`
            Número máximo de épocas
        batch_size : int, optional
            Tamaño del mini-batch. `None` usa el dataset completo.
        lr_schedule : {'linear', 'exponential'}, optional
            Esquema de scheduling del learning rate. `None` desactiva el scheduling.
        lr_min : float, default `1e-5`
            Learning rate mínimo para scheduling lineal. Llamada "saturación".
        gamma : float, default `0.99`
            Factor de decaimiento por época para scheduling exponencial.
        patience : int, optional
            Épocas sin mejora en val_loss antes de early stopping. Requiere datos de validación.
        verbose : bool, default `True`
            Muestra barra de progreso e información del entrenamiento.
        '''
        n_classes = self.layers[-1][0]
        Y_train = to_onehot(y_train, n_classes)
        m_train = X_train.shape[0]
        bs = m_train if batch_size is None else batch_size
        val_history = (X_val is not None and y_val is not None)
        if val_history:
            Y_val = to_onehot(y_val, n_classes)

        self.optimizer.setup(self.weights, self.biases)
        lr0 = self.optimizer.lr

        history = {'train_loss': []}
        if val_history: history['val_loss'] = []
        if lr_schedule: history['lr'] = []

        best_val_loss = np.inf
        epochs_no_improve = 0
        t0 = time.time()

        epochs_list = range(1, epochs + 1)
        bar = tqdm(epochs_list, desc='Training', unit='epoch', colour='blue') if verbose else None
        iterator = bar if verbose else epochs_list

        for epoch in iterator:
            if lr_schedule == 'linear':
                self.optimizer.lr = max(lr0 * (1 - epoch / epochs), lr_min)
            elif lr_schedule == 'exponential':
                self.optimizer.lr = lr0 * (gamma ** epoch)

            idx = np.random.permutation(m_train)
            X_sh, Y_sh = X_train[idx], Y_train[idx]

            for start in range(0, m_train, bs): 
                self.forward(X_sh[start:start + bs])
                self.backward(Y_sh[start:start + bs])
                self.update()

            train_loss = cross_entropy(self.forward(X_train), Y_train)
            history['train_loss'].append(train_loss)

            if val_history:
                val_loss = cross_entropy(self.forward(X_val), Y_val)
                history['val_loss'].append(val_loss)

            if lr_schedule:
                history['lr'].append(self.optimizer.lr)

            if verbose:
                post = dict(train_loss=f'{train_loss:.4f}')
                if val_history: post['val_loss'] = f'{val_loss:.4f}'
                bar.set_postfix(**post)

            if patience is not None and val_history:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            bar.colour = 'green'
                            bar.refresh()
                        break

        if verbose and epochs_no_improve < (patience or np.inf):
            bar.colour = 'yellow'
            bar.refresh()

        self.optimizer.lr = lr0

        total_time = time.time() - t0
        stopped_early = patience is not None and val_history and epochs_no_improve >= patience
        status = 'early stopping' if stopped_early else 'completed'

        if verbose:
            parts = [f'epoch: {epoch}', f'train_loss: {history["train_loss"][-1]:.4f}']
            if val_history: parts.append(f'val_loss: {history["val_loss"][-1]:.4f}')
            parts.append(f'time: {total_time:.1f}s')
            print(f'[{status}] — {" | ".join(parts)}')

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)
