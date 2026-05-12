import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

from src.models import NN
from src.utils import to_onehot, LABELS, LABELS_PLUS_IDX
from src.metrics import accuracy, cross_entropy, confusion_matrix, f1_macro

def show_images(X : np.ndarray, y : np.ndarray):
    _, axes = plt.subplots(3, 5, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(axes.size):
        idx = np.random.randint(0, X.shape[0])
        img = X[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{LABELS[y[idx]]} ({y[idx]})')
        axes[i].axis('off')
    plt.suptitle('Samples Visualization', fontsize=16, y=1)
    plt.tight_layout()
    plt.show()

def labels_distribution(y : np.ndarray):
    plt.figure(figsize=(12,4))
    sbn.histplot(
        y,
        stat='density',
        alpha=0.3,
        kde=True,
        bins=49
        )
    plt.xticks(range(49), LABELS)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution', fontsize=12)
    plt.show()

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
    plt.legend()
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
    b_val = ax.bar(x + width/2, val_vals,   width, label='Validation', color='tab:green')
    ax.bar_label(b_train, fmt='%.4f', padding=4)
    ax.bar_label(b_val,   fmt='%.4f', padding=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(train_vals), max(val_vals)) * 1.1)
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
    cm_val = confusion_matrix(yhat_val, y_val, n_classes)

    _, axes = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'height_ratios': [20, 1]})

    for col, (cm, title) in enumerate(zip([cm_train, cm_val], ['Train', 'Validation'])):
        ax = axes[0, col]
        cbar_ax = axes[1, col]
        sbn.heatmap(
            cm, ax=ax, annot=False, cmap='Blues', vmin=0, vmax=cm.max(),
            xticklabels=LABELS_PLUS_IDX, yticklabels=LABELS_PLUS_IDX,
            cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'}
                    )
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle('Confusion Matrix Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()