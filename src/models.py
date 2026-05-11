import numpy as np
from src.activations import ACTIVATIONS
from src.utils import to_onehot
from src.metrics import cross_entropy

class NN:
    '''
    ***Neural Network*** / ***Multi-layer Perceptron*** para clasificación multiclase
    - Capa de salida: Softmax
    - Función de pérdida: Entropía Cruzada
    '''
    def __init__(self, input_size : int, layers : list[tuple]):
        """
        Parameters
        ----------
        input_size : int
            Tamaño del vector input `x`
        layers: list[tuple]
            Lista de tuplas `(n_neurons, activation_name)` representando cada **capa oculta**
        """
        self.layers = layers
        self.weights = []
        self.biases  = []

        # He initialization (sigma^2 = 2/n_in)
        in_size = input_size
        for (out_size, _) in layers:
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
            Matriz o vector resultante de la capa de salida (one-hot)

        Note
        ----
        Algoritmo basado en el Algoritmo 8.1 de "Deep Learning: Foundations and Concepts -- C.M & H Bishop"
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

    def backward(self, y_onehot):
        '''
        ***Backward Pass***

        Computa y almacena los gradientes de la pérdida respecto a los pesos y biases.
        No actualiza los parámetros — eso es responsabilidad del optimizador.

        Note
        ----
        Algoritmo basado en el Algoritmo 8.1 de "Deep Learning: Foundations and Concepts -- C.M & H Bishop"
        '''
        m = y_onehot.shape[0]
        L = len(self.layers)
        deltas = [None] * L

        # Output layer: softmax + cross entropy Jacobian -> z^(L) - y
        deltas[L - 1] = self.z_cache[L] - y_onehot

        # Hidden layers: δ^(l) = φ'(a^(l)) ⊙ (W^(l+1)ᵀ δ^(l+1))
        for l in range(L - 2, -1, -1):
            _, act_deriv = ACTIVATIONS[self.layers[l][1]]
            deltas[l] = (deltas[l + 1] @ self.weights[l + 1].T) * act_deriv(self.a_cache[l])

        self.dW = [self.z_cache[l].T @ deltas[l] / m for l in range(L)]
        self.db = [np.sum(deltas[l], axis=0, keepdims=True) / m for l in range(L)]

    def update(self, lr):
        '''
        ***Gradient Descent Step***

        Actualiza los pesos y biases usando los gradientes calculados en `backward`.
        '''
        for l in range(len(self.layers)):
            self.weights[l] -= lr * self.dW[l]
            self.biases[l] -= lr * self.db[l]

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr):
        '''
        ***Training***

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Datos de entrenamiento
        X_val, y_val : np.ndarray, optional
            Datos de validación (para obtener valores de loss en cada época de ser necesario) 
        epochs : int
            número máximo de épocas
        eta : float
            Learning rate
        '''
        n_classes = self.layers[-1][0]
        Y_train = to_onehot(y_train, n_classes)
        Y_val = to_onehot(y_val, n_classes)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            y_hat_train = self.forward(X_train)
            self.backward(Y_train)
            self.update(lr)

            train_loss = cross_entropy(y_hat_train, Y_train)
            val_loss = cross_entropy(self.forward(X_val), Y_val)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch:>4d}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}')

        return history
