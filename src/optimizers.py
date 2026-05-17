import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Clase base abstracta para optimizadores de gradiente."""

    @abstractmethod
    def setup(self, weights, biases, *args, **kwargs):
        """Inicializa el estado interno del optimizador (momentos, acumuladores, etc.)."""
        pass

    @abstractmethod
    def step(self, weights, biases, dW, db, *args, **kwargs):
        """Aplica una actualización de parámetros dado el gradiente actual."""
        pass

class GD(Optimizer):
    """
    Gradient Descent con weight decay (L2 regularización sobre pesos).

    Parameters
    ----------
    lr : float, default `0.1`
        Learning rate.
    weight_decay : float, default `0.0`
        Factor de regularización L2. `0` desactiva el weight decay.
    """

    def __init__(self, lr: float = 0.1, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def setup(self, weights, biases):
        pass

    def step(self, weights, biases, dW, db):
        for l in range(len(weights)):
            weights[l] = (1 - self.lr * self.weight_decay) * weights[l] - self.lr * dW[l]
            biases[l] -= self.lr * db[l]

SGD = GD # solo cambia en batch_size de los modelos. 
# batch_size=1    -> SGD
# batch_size=None -> GD
# batch_size=k    -> mini-batch SGD

class Momentum(Optimizer):
    """
    Gradient Descent con Momentum.

    Acumula un promedio exponencial de gradientes pasados para suavizar
    las actualizaciones y acelerar la convergencia.

    Parameters
    ----------
    lr : float, default `0.01`
        Learning rate.
    beta : float, default `0.9`
        Factor de decaimiento del momento. Valores cercanos a 1 dan más peso a los gradientes pasados.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta

    def setup(self, weights, biases):
        self._vW = [np.zeros_like(w) for w in weights]
        self._vb = [np.zeros_like(b) for b in biases]

    def step(self, weights, biases, dW, db):
        for l in range(len(weights)):
            self._vW[l] = self.beta * self._vW[l] + self.lr * dW[l]
            self._vb[l] = self.beta * self._vb[l] + self.lr * db[l]
            weights[l] -= self._vW[l]
            biases[l]  -= self._vb[l]

class AdaGrad(Optimizer):
    """
    Adaptive Gradient (AdaGrad).

    Adapta el learning rate por parámetro acumulando la suma de cuadrados
    de los gradientes históricos, reduciendo el lr para parámetros frecuentes
    y manteniéndolo alto para parámetros poco actualizados.

    Parameters
    ----------
    lr : float, default `1e-2`
        Learning rate global.
    eps : float, default `1e-8`
        Término de estabilidad numérica para evitar división por cero.
    """
    def __init__(self, lr: float = 1e-2, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps

    def setup(self, weights, biases):
        self._GW = [np.zeros_like(w) for w in weights]
        self._Gb = [np.zeros_like(b) for b in biases]

    def step(self, weights, biases, dW, db):
        for l in range(len(weights)):
            for param, grad, G in [
                (weights, dW, self._GW),
                (biases,  db, self._Gb),
            ]:
                G[l] += grad[l] ** 2
                param[l] -= self.lr * grad[l] / (np.sqrt(G[l]) + self.eps)

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam).

    Combina Momentum (primer momento) y segundo momento del gradiente con
    corrección de sesgo para las estimaciones iniciales.

    Parameters
    ----------
    lr : float, default `1e-3`
        Learning rate.
    beta1 : float, default `0.9`
        Factor de decaimiento del primer momento (media de gradientes).
    beta2 : float, default `0.999`
        Factor de decaimiento del segundo momento (varianza de gradientes).
    eps : float, default `1e-8`
        Término de estabilidad numérica para evitar división por cero.
    """
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def setup(self, weights, biases):
        self._t  = 0
        self._mW = [np.zeros_like(w) for w in weights]
        self._mb = [np.zeros_like(b) for b in biases]
        self._vW = [np.zeros_like(w) for w in weights]
        self._vb = [np.zeros_like(b) for b in biases]

    def step(self, weights, biases, dW, db):
        self._t += 1
        for l in range(len(weights)):
            for param, grad, m, v in [
                (weights, dW, self._mW, self._vW),
                (biases,  db, self._mb, self._vb),
            ]:
                m[l] = self.beta1 * m[l] + (1 - self.beta1) * grad[l]
                v[l] = self.beta2 * v[l] + (1 - self.beta2) * grad[l] ** 2
                m_hat = m[l] / (1 - self.beta1 ** self._t)
                v_hat = v[l] / (1 - self.beta2 ** self._t)
                param[l] = param[l] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class AdamW(Optimizer):
    """
    Adam con Weight Decay desacoplado (AdamW).

    Aplica la regularización L2 directamente sobre los pesos antes de la
    actualización de Adam, en lugar de incorporarla al gradiente como en Adam+L2.
    Esto evita que el weight decay interactúe con la adaptación del learning rate.

    Parameters
    ----------
    lr : float, default `1e-3`
        Learning rate.
    beta1 : float, default `0.9`
        Factor de decaimiento del primer momento.
    beta2 : float, default `0.999`
        Factor de decaimiento del segundo momento.
    eps : float, default `1e-8`
        Término de estabilidad numérica.
    weight_decay : float, default `0.0`
        Factor de regularización L2 desacoplado. `0` equivale a Adam estándar.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def setup(self, weights, biases):
        self._t  = 0
        self._mW = [np.zeros_like(w) for w in weights]
        self._mb = [np.zeros_like(b) for b in biases]
        self._vW = [np.zeros_like(w) for w in weights]
        self._vb = [np.zeros_like(b) for b in biases]

    def step(self, weights, biases, dW, db):
        self._t += 1
        for l in range(len(weights)):
            for param, grad, m, v in [
                (weights, dW, self._mW, self._vW),
                (biases, db, self._mb, self._vb),
            ]:
                m[l] = self.beta1 * m[l] + (1 - self.beta1) * grad[l]
                v[l] = self.beta2 * v[l] + (1 - self.beta2) * grad[l] ** 2
                m_hat = m[l] / (1 - self.beta1 ** self._t)
                v_hat = v[l] / (1 - self.beta2 ** self._t)
                param[l] = ((1 - self.lr * self.weight_decay) * param[l]
                            - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
