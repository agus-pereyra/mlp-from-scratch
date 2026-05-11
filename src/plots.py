import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

from src.models import NN
from src.utils import to_onehot
from src.metrics import accuracy, cross_entropy, confusion_matrix, f1_macro

def loss_history(output : dict):
    epochs = len(output['train_loss'])
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), output['train_loss'], 'tab:blue', label='Train Loss')
    plt.plot(range(1, epochs+1), output['val_loss'], 'tab:green', label='Validation Loss')
    plt.xlim(left=1, right=epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function: Cross Entropy')
    plt.title('Training Loss History', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.show()

def compare_metrics(model : NN, X_train, y_train, X_val, y_val):
    n_classes = model.layers[-1][0]
    y_train_oh = to_onehot(y_train, n_classes)
    y_val_oh = to_onehot(y_val,   n_classes)

    yhat_train = model.forward(X_train)
    yhat_val = model.forward(X_val)

    metrics = {
        'Accuracy': [accuracy(yhat_train, y_train), accuracy(yhat_val, y_val)],
        'Cross Entropy': [cross_entropy(yhat_train, y_train_oh), cross_entropy(yhat_val, y_val_oh)],
        'F1 Macro': [f1_macro(yhat_train, y_train, n_classes), f1_macro(yhat_val, y_val, n_classes)],
    }

    labels = list(metrics.keys())
    train_vals = [v[0] for v in metrics.values()]
    val_vals = [v[1] for v in metrics.values()]
    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(8, 5))
    b_train = ax.bar(x - width/2, train_vals, width, label='Train', color='tab:blue')
    b_val = ax.bar(x + width/2, val_vals,   width, label='Validation', color='tab:orange')
    ax.bar_label(b_train, fmt='%.4f', padding=4)
    ax.bar_label(b_val,   fmt='%.4f', padding=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(train_vals), max(val_vals)) * 1.25)
    ax.set_ylabel('Score')
    ax.set_title('Train vs Validation Metrics')
    ax.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def compare_confusion_matrix(model : NN, X_train, y_train, X_val, y_val):
    n_classes = model.layers[-1][0]
    yhat_train = model.forward(X_train)
    yhat_val = model.forward(X_val)
    cm_train = confusion_matrix(yhat_train, y_train, n_classes)
    cm_val = confusion_matrix(yhat_val,   y_val,   n_classes)

    _, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, cm, title in zip(axes, [cm_train, cm_val], ['Train', 'Validation']):
        sbn.heatmap(cm, ax=ax, annot=False, cmap='Blues',
                    xticklabels=range(n_classes), yticklabels=range(n_classes))
        ax.set_title(f'{title}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.suptitle('Confusion Matrix Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()