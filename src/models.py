import time
import numpy as np
from tqdm import tqdm
from typing import Literal
from copy import deepcopy
from src.activations import ACTIVATIONS
from src.utils import to_onehot
from src.metrics import cross_entropy

class NN:
    '''
    ***Neural Network*** / ***Multi-layer Perceptron*** para clasificación multiclase
    - Capa de salida: Softmax
    - Función de pérdida: Entropía Cruzada
    '''
    def __init__(self, input_size : int, output_size : int, layers : list[tuple]):
        """
        Parameters
        ----------
        input_size : int
            Tamaño del vector input `x`
        output_size : int
            Tamaño del vector output `y`
        layers: list[tuple]
            Lista de tuplas `(n_neurons, activation_name)` representando cada **capa oculta**
        """
        self.layers = deepcopy(layers)
        self.layers.append((output_size, 'softmax'))
        self.weights = []
        self.biases  = []

        # He initialization (sigma^2 = 2/n_in)
        in_size = input_size
        for (out_size, _) in self.layers:
            self.weights.append(np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size))
            self.biases.append(np.zeros((1, out_size)))
            in_size = out_size

    def forward(self, X : np.ndarray):
        '''
        ***Forward Pass***

        Cálculo de pre-activaciones (a) y activaciones (z) de la red para los datos X 
        y los pesos actuales del modelo. 

        Parameter
        ----------
        X : np.ndarray
            Matriz o vector de datos

        Return
        ------
        y_hat : np.ndarray
            Matriz o vector one-hot resultante de la capa de salida (última activación == z^(L))

        Note
        ----
        Algoritmo basado en el Algoritmo 8.1 de "Deep Learning: Foundations and Concepts -- C.M & H Bishop"
        adaptado de forma matricial.
        '''
        self.z_cache = [X]  # activations: z = phi(a),  z[0] = x
        self.a_cache = []   # pre-activations: a = Wz + b

        z = X
        for i, (_, act_name) in enumerate(self.layers):
            a = z @ self.weights[i] + self.biases[i]
            phi, _ = ACTIVATIONS[act_name]
            z = phi(a)
            self.a_cache.append(a)
            self.z_cache.append(z)

        return z

    def backward(self, y_onehot : np.ndarray):
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

        deltas[L - 1] = self.z_cache[L] - y_onehot # output layer

        for l in range(L - 2, -1, -1):             # hidden layers
            _, act_deriv = ACTIVATIONS[self.layers[l][1]]
            deltas[l] = (deltas[l + 1] @ self.weights[l + 1].T) * act_deriv(self.a_cache[l])

        self.dW = [self.z_cache[l].T @ deltas[l] / m for l in range(L)]
        self.db = [np.sum(deltas[l], axis=0, keepdims=True) / m for l in range(L)]

    def update(self, lr : float = 0.1):
        '''
        ***Gradient Descent Step***

        Actualiza los pesos y biases usando los gradientes calculados en `backward`.

        Parameter
        ---------
        lr : float, default `0.1`
            Learning Rate
        '''
        for l in range(len(self.layers)):
            self.weights[l] -= lr * self.dW[l]
            self.biases[l] -= lr * self.db[l]

    def fit(
            self,
            X_train : np.ndarray,
            y_train : np.ndarray,
            X_val : np.ndarray = None,
            y_val : np.ndarray = None,
            epochs : int = 500,
            lr : float = 0.1,
            verbose : bool = True
            ):
        '''
        ***Entrenamiento*** de la red a partir de iteraciones back-propagation para cálculo del gradiente
        y Gradient-Descent estándar para optimización de los pesos.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Datos de entrenamiento
        X_val, y_val : np.ndarray, default `None`, optional
            Datos de validación, necesarios solo en el caso de requerir historial de loss para validation.
        epochs : int, default `500`
            Número de épocas
        lr : float, default `0.1`
            Learning rate
        verbose : bool, default `True`
            Muestra en pantalla una barra de progreso con la información del entrenamiento
        '''
        n_classes = self.layers[-1][0]
        Y_train = to_onehot(y_train, n_classes)
        Y_val = to_onehot(y_val, n_classes)

        val_history = (X_val is not None and y_val is not None)

        history = {'train_loss': [], 'val_loss': []} if val_history else {'train_loss' : []}
        t0 = time.time()

        epochs_list = range(1, epochs + 1)
        bar = tqdm(epochs_list, desc='Training', unit='epoch', colour='blue')
        iterator = bar if verbose else epochs_list

        for epoch in iterator:
            y_hat_train = self.forward(X_train)
            self.backward(Y_train)
            self.update(lr)

            train_loss = cross_entropy(y_hat_train, Y_train)
            if val_history: val_loss = cross_entropy(self.forward(X_val), Y_val)

            history['train_loss'].append(train_loss)
            if val_history: history['val_loss'].append(val_loss)

            if verbose:
                if val_history:
                    bar.set_postfix(epoch=epoch, train_loss=f'{train_loss:.4f}', val_loss=f'{val_loss:.4f}')
                else:
                    bar.set_postfix(epoch=epoch, train_loss=f'{train_loss:.4f}')

        if verbose:
            bar.colour = 'green'
            bar.refresh()

        total_time = time.time() - t0
        if verbose:
            if val_history: 
                print(f'\nFinal -- epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | val_loss: {history["val_loss"][-1]:.4f} | time: {total_time:.1f}s')
            else:
                print(f'\nFinal -- epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | time: {total_time:.1f}s')

        return history
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        '''
        Realiza una predicción cuyo resultado es el label de la clase predicha

        Parameter
        ----------
        X : np.ndarray
            Matriz o vector de datos

        Return
        ------
        y_hat : np.ndarray
            Vector de etiqueta(s) de predicciones.
        '''
        return np.argmax(self.forward(X), axis=1)

class AdvancedNN(NN):
    def __init__(self, input_size : int, output_size : int, layers : list[tuple]):
        super().__init__(input_size, output_size, layers)

    def update(self, lr : float, beta1 : float = 0.9, beta2 : float = 0.999, eps : float = 1e-8, weight_decay : float = 0.0):
        '''
        ***AdamW Step***

        Actualiza los pesos y biases usando los momentos calculados en `backward`,
        con weight decay desacoplado aplicado después del paso de Adam.

        Parameters
        ----------
        lr : float
            Learning rate
        beta1, beta2 : float, defaults `0.9` y `0.999`
            Decay rates para el primer y segundo momento
        eps : float, default `1e-8`
            Término de estabilidad numérica
        weight_decay : float, default `0.0`
            Factor de decaimiento de pesos λ. `0.0` desactiva el weight decay.
        '''
        self._t += 1
        for l in range(len(self.layers)):
            for param, grad, m, v in [
                (self.weights, self.dW, self._mW, self._vW),
                (self.biases,  self.db, self._mb, self._vb),
            ]:
                m[l] = beta1 * m[l] + (1 - beta1) * grad[l]
                v[l] = beta2 * v[l] + (1 - beta2) * grad[l] ** 2
                m_hat = m[l] / (1 - beta1 ** self._t)
                v_hat = v[l] / (1 - beta2 ** self._t)
                param[l] = (1 - lr * weight_decay) * param[l] - lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
            self,
            X_train : np.ndarray,
            y_train : np.ndarray,
            X_val : np.ndarray = None,
            y_val : np.ndarray = None,
            epochs : int = 500,
            lr : float = 1e-3,
            batch_size : int = None,
            beta1 : float = 0.9,
            beta2 : float = 0.999,
            eps : float = 1e-8,
            lr_schedule : Literal['linear', 'exponential'] = None,
            lr_min : float = 1e-5,
            gamma : float = 0.99,
            weight_decay : float = 0.0,
            patience : int = None,
            verbose : bool = True
            ):
        '''
        ***Entrenamiento*** de la red a partir de iteraciones back-propagation para cálculo del gradiente
        y Adam para optimización de los pesos, con soporte de mini-batches y learning rate scheduling.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Datos de entrenamiento
        X_val, y_val : np.ndarray, default `None`, optional
            Datos de validación, necesarios solo en el caso de requerir early stopping y/o historial de loss para validation.
        epochs : int
            Número máximo de épocas
        lr : float
            Learning rate inicial
        batch_size : int, optional
            Tamaño del mini-batch. `None` usa el dataset completo (batch gradient descent).
        beta1, beta2 : float
            Decay rates del primer y segundo momento de Adam
        eps : float
            Término de estabilidad numérica de Adam
        lr_schedule : {'linear', 'exponential'}, optional
            Esquema de scheduling. `None` desactiva el scheduling.
        lr_min : float
            Tasa de aprendizaje mínima (piso) para scheduling lineal.
        gamma : float
            Factor de decaimiento por época para scheduling exponencial.
        weight_decay : float, default `0.0`
            Factor lambda de weight decay desacoplado (AdamW). El default es sin regularización.
        patience : int, optional
            Épocas sin mejora en val_loss antes de detener el entrenamiento. `None` desactiva early stopping.
            Requiere que `X_val` y `y_val` sean provistos.
        verbose : bool
            Muestra en pantalla una barra de progreso con la información del entrenamiento
        '''
        n_classes = self.layers[-1][0]
        Y_train = to_onehot(y_train, n_classes)
        m_train = X_train.shape[0]
        bs = m_train if batch_size is None else batch_size
        val_history = (X_val is not None and y_val is not None)
        if val_history: Y_val = to_onehot(y_val, n_classes)

        self._t  = 0
        # momentos acumulados de adam (m y v por cada capa y matriz de pesos + vector de bias)
        # momentos de 1er orden
        self._mW = [np.zeros_like(w) for w in self.weights] 
        self._mb = [np.zeros_like(b) for b in self.biases]
        # momentos de 2do orden
        self._vW = [np.zeros_like(w) for w in self.weights] 
        self._vb = [np.zeros_like(b) for b in self.biases]

        history = {'train_loss': [], 'lr': []}
        if val_history: history['val_loss'] = []
        t0 = time.time()
        lr0 = lr

        best_val_loss = np.inf
        epochs_no_improve = 0

        epochs_list = range(1, epochs + 1)
        if verbose:
            bar = tqdm(epochs_list, desc='Training', unit='epoch', colour='blue')
            iterator = bar
        else:
            bar = None
            iterator = epochs_list

        for epoch in iterator:
            if lr_schedule == 'linear':
                lr = max(lr0 * (1 - epoch / epochs), lr_min)
            elif lr_schedule == 'exponential':
                lr = lr0 * (gamma ** epoch)

            idx = np.random.permutation(m_train)
            X_shuffled, Y_shuffled = X_train[idx], Y_train[idx]

            for start in range(0, m_train, bs):
                X_batch = X_shuffled[start:start + bs]
                Y_batch = Y_shuffled[start:start + bs]
                self.forward(X_batch)
                self.backward(Y_batch)
                self.update(lr, beta1, beta2, eps, weight_decay)

            train_loss = cross_entropy(self.forward(X_train), Y_train)
            if val_history: val_loss = cross_entropy(self.forward(X_val), Y_val)

            history['train_loss'].append(train_loss)
            if val_history: history['val_loss'].append(val_loss)
            history['lr'].append(lr)

            if verbose:
                if val_history:
                    bar.set_postfix(epoch=epoch, train_loss=f'{train_loss:.4f}', val_loss=f'{val_loss:.4f}')
                else:
                    bar.set_postfix(epoch=epoch, train_loss=f'{train_loss:.4f}')

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

        total_time = time.time() - t0
        stopped_early = patience is not None and val_history and epochs_no_improve >= patience
        status = 'early stopping' if stopped_early else 'completed'
        if verbose:
            if val_history:
                print(f'\nFinal [{status}] — epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | val_loss: {history["val_loss"][-1]:.4f} | time: {total_time:.1f}s')
            else:
                print(f'\nFinal [{status}] — epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | time: {total_time:.1f}s')

        return history