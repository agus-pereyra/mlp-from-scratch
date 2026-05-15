import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def setup(self, weights, biases, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, weights, biases, dW, db, *args, **kwargs):
        pass

class GD(Optimizer):
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def setup(self, weights, biases):
        pass

    def step(self, weights, biases, dW, db):
        for l in range(len(weights)):
            weights[l] -= self.lr * dW[l]
            biases[l] -= self.lr * db[l]

SGD = GD

class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
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
                (biases,  db, self._mb, self._vb),
            ]:
                m[l] = self.beta1 * m[l] + (1 - self.beta1) * grad[l]
                v[l] = self.beta2 * v[l] + (1 - self.beta2) * grad[l] ** 2
                m_hat = m[l] / (1 - self.beta1 ** self._t)
                v_hat = v[l] / (1 - self.beta2 ** self._t)
                param[l] = ((1 - self.lr * self.weight_decay) * param[l]
                            - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
