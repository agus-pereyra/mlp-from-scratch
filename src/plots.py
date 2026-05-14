import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sbn

from src.models import NN
from src.utils import to_onehot, LABELS, LABELS_IDX_X, LABELS_IDX_Y
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
    plt.xticks(range(49), LABELS_IDX_X)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution', fontsize=12)
    plt.show()

def loss_history(output: dict, ax: plt.Axes = None):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 5))

    epochs = len(output['train_loss'])
    ax.plot(range(1, epochs+1), output['train_loss'], 'tab:blue',  label='Train Loss')
    ax.plot(range(1, epochs+1), output['val_loss'],   'tab:green', label='Validation Loss')
    ax.set_xlim(left=1, right=epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross Entropy')
    ax.set_title('Training Loss History', fontsize=14)
    ax.grid()
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()

def lr_history(output: dict, ax: plt.Axes = None):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 5))

    values = output['lr']
    epochs = len(values)
    ax.plot(range(1, epochs+1), values, 'tab:orange', label='Learning Rate')
    ax.set_xlim(left=1, right=epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid()
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()

def compare_metrics(model: NN, X_train, y_train, X_val, y_val, ax: plt.Axes = None):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 5))

    n_classes = model.layers[-1][0]
    y_train_oh = to_onehot(y_train, n_classes)
    y_val_oh = to_onehot(y_val, n_classes)

    yhat_train = model.forward(X_train)
    yhat_val = model.forward(X_val)

    metrics = {
        'Accuracy' : [accuracy(yhat_train, y_train), accuracy(yhat_val, y_val)],
        'Cross Entropy' :[cross_entropy(yhat_train, y_train_oh), cross_entropy(yhat_val, y_val_oh)],
        'F1 Macro' : [f1_macro(yhat_train, y_train, n_classes), f1_macro(yhat_val, y_val, n_classes)],
    }

    labels = list(metrics.keys())
    train_vals = [v[0] for v in metrics.values()]
    val_vals = [v[1] for v in metrics.values()]
    x = np.arange(len(labels))
    width = 0.35

    b_train = ax.bar(x - width/2, train_vals, width, label='Train', color='tab:blue')
    b_val = ax.bar(x + width/2, val_vals, width, label='Validation', color='tab:green')
    ax.bar_label(b_train, fmt='%.4f', padding=4)
    ax.bar_label(b_val, fmt='%.4f', padding=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(train_vals), max(val_vals)) * 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Train vs Validation Metrics')
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    if standalone:
        plt.tight_layout()
        plt.show()

def compare_confusion_matrix(model: NN, X_train, y_train, X_val, y_val, axes=None):
    standalone = axes is None
    if standalone:
        _, axes = plt.subplots(
            2, 2, figsize=(16, 9),
            gridspec_kw={'height_ratios': [20, 1], 'hspace': 0.45}
            )

    n_classes = model.layers[-1][0]
    yhat_train = model.forward(X_train)
    yhat_val = model.forward(X_val)
    cm_train = confusion_matrix(yhat_train, y_train, n_classes)
    cm_val = confusion_matrix(yhat_val, y_val, n_classes)

    for col, (cm, title, cmap) in enumerate(zip(
                [cm_train, cm_val], ['Train', 'Validation'], ['Blues', 'Greens']
                )):
        ax = axes[0, col]
        cbar_ax = axes[1, col]
        sbn.heatmap(
            cm, ax=ax, annot=False, cmap=cmap, vmin=0, vmax=cm.max(),
            xticklabels=LABELS_IDX_X, yticklabels=LABELS_IDX_Y if col == 0 else False,
            cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        if col == 0:
            ax.set_ylabel('True')

    if standalone:
        plt.suptitle('Confusion Matrix Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

def training_summary(model: NN, train_hist: dict, X_train, y_train, X_val, y_val, title: str = None):
    """
    Summary figure with all training plots in one view.
    lr_history is included only if 'lr' is present in train_hist.
    """
    has_lr = 'lr' in train_hist and len(train_hist['lr']) > 0

    n_top_cols = 3 if has_lr else 2
    fig = plt.figure(figsize=(18, 14))

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2.2], hspace=0.2)

    top_gs = gridspec.GridSpecFromSubplotSpec(1, n_top_cols, subplot_spec=outer[0], wspace=0.35)
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1],
        height_ratios=[20, 0.5],
        wspace=0.05, hspace=0.3
        )

    ax_loss = fig.add_subplot(top_gs[0])
    ax_metrics = fig.add_subplot(top_gs[-1])
    ax_lr = fig.add_subplot(top_gs[1]) if has_lr else None

    cm_axes = np.array([
        [fig.add_subplot(bot_gs[0, 0]), fig.add_subplot(bot_gs[0, 1])],
        [fig.add_subplot(bot_gs[1, 0]), fig.add_subplot(bot_gs[1, 1])],
    ])

    loss_history(train_hist, ax=ax_loss)
    if has_lr:
        lr_history(train_hist, ax=ax_lr)
    compare_metrics(model, X_train, y_train, X_val, y_val, ax=ax_metrics)
    compare_confusion_matrix(model, X_train, y_train, X_val, y_val, axes=cm_axes)

    fig.suptitle(title or 'Training Summary', fontsize=20, y=0.93)
    plt.show()
