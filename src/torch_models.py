import time
import numpy as np
import torch
import torch.nn as nn
from typing import Type
from tqdm import tqdm

TORCH_ACTIVATIONS = {
    'relu' : nn.ReLU,
    'leaky_relu' : nn.LeakyReLU,
    'silu' : nn.SiLU,
    'swish' : nn.SiLU, # swish == silu
    'gelu' : nn.GELU,
    'linear': nn.Identity,
}

class TorchNN(nn.Module):
    """
    MLP implementada en **PyTorch** con la misma interfaz que `NN`.
    """
    def __init__(
            self, input_size: int, output_size: int, layers: list[tuple],
            dropout: float = 0.0,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam, 
            optim_params: dict = None
            ):
        """
        Parameters
        ----------
        input_size : int
            Tamaño del vector input `x`
        output_size : int
            Tamaño del vector output `y`
        layers : list[tuple]
            Lista de tuplas (n_neurons, activation_name) para las capas ocultas.
        dropout : float
            Probabilidad de dropout aplicado después de cada capa oculta (0 = desactivado).
        """
        super().__init__()
        self.layers = list(layers) + [(output_size, 'softmax')]

        modules = []
        in_size = input_size
        for out_size, act_name in layers: # cada capa es [Linear -> Phi -> Drouput]
            modules.append(nn.Linear(in_size, out_size))
            modules.append(TORCH_ACTIVATIONS[act_name]())
            if dropout > 0.0:
                modules.append(nn.Dropout(p=dropout))
            in_size = out_size
        modules.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*modules)
        self.optimizer  = optimizer
        self.optim_params = optim_params if optim_params is not None else {}
        self._param_init()

    def _param_init(self):
        '''
        ***He initialization***
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Devuelve Scores crudos de la predicción (output de la red antes de pasa por softmax).
        '''
        return self.network(x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve probabilidades softmax como np.ndarray (para compatibilidad con métricas).
        No se utiliza en optimización.
        """
        device = next(self.parameters()).device
        x = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self._logits(x), dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def fit(
            self,
            X_train : np.ndarray,
            y_train : np.ndarray,
            X_val : np.ndarray = None,
            y_val : np.ndarray = None,
            epochs : int = 200,
            batch_size : int = None,
            lr_schedule : str = None,
            gamma : float = 0.99,
            lr_min : float = 1e-5,
            patience : int = None,
            verbose : bool = True,
        ) -> dict:
        """
        Entrena la red con mini-batches, lr scheduling y early stopping.
        El optimizador se configura en `__init__` via `optimizer` y `optim_params`.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(device)
        val_history = X_val is not None and y_val is not None
        if val_history:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(device)

        n = X_t.shape[0]
        bs = n if batch_size is None else batch_size

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = self.optimizer(self.parameters(), **self.optim_params)
        lr = optimizer.param_groups[0]['lr']

        if lr_schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif lr_schedule == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=lr_min / lr, total_iters=epochs
            )
        else:
            scheduler = None

        history = {'train_loss': []}
        if val_history: history['val_loss'] = []
        if lr_schedule: history['lr'] = []

        best_val_loss = np.inf
        epochs_no_improve = 0
        t0 = time.time()

        epochs_list = range(1, epochs + 1)
        if verbose:
            bar = tqdm(epochs_list, desc='Training', unit='epoch', colour='blue')
            iterator = bar
        else:
            bar = None
            iterator = epochs_list

        for epoch in iterator:
            self.train()
            idx = torch.randperm(n)
            X_shuf, y_shuf = X_t[idx], y_t[idx]

            for start in range(0, n, bs):
                xb, yb = X_shuf[start:start + bs], y_shuf[start:start + bs]
                optimizer.zero_grad()
                criterion(self._logits(xb), yb).backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                train_loss = criterion(self._logits(X_t), y_t).item()
                if val_history:
                    val_loss = criterion(self._logits(X_v), y_v).item()

            history['train_loss'].append(train_loss)
            if val_history:  history['val_loss'].append(val_loss)
            if lr_schedule:  history['lr'].append(optimizer.param_groups[0]['lr'])

            if scheduler:
                scheduler.step()

            if verbose:
                postfix = dict(train_loss=f'{train_loss:.4f}')
                if val_history:
                    postfix['val_loss'] = f'{val_loss:.4f}'
                bar.set_postfix(**postfix)

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

        if verbose and bar is not None:
            stopped_early = patience is not None and val_history and epochs_no_improve >= patience
            if not stopped_early:
                bar.colour = 'yellow'
                bar.refresh()
            status = 'early stopping' if stopped_early else 'completed'
            total_time = time.time() - t0
            if val_history:
                print(f'[{status}] — epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | val_loss: {history["val_loss"][-1]:.4f} | time: {total_time:.1f}s')
            else:
                print(f'[{status}] — epoch: {epoch} | train_loss: {history["train_loss"][-1]:.4f} | time: {total_time:.1f}s')

        return history

