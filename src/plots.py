import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sbn

from src.models import NN
from src.utils import to_onehot, LABELS, LABELS_IDX_X, LABELS_IDX_Y
from src.metrics import accuracy, cross_entropy, confusion_matrix, f1_macro

def show_images(X : np.ndarray, y : np.ndarray):
    np.random.seed(36631)
    classes = np.unique(y)
    n_cols = 7
    n_rows = int(np.ceil(len(classes) / n_cols))
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    axes = axes.flatten()
    for ax, c in zip(axes, classes):
        idx = np.random.choice(np.where(y == c)[0])
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'{LABELS[c]} ({c})', fontsize=11)
        ax.axis('off')
    plt.suptitle('Samples Visualization (one per class)', fontsize=20, y=0.99)
    plt.tight_layout()
    plt.show()

def show_class_samples(X : np.ndarray, y : np.ndarray, labels : list[int], n : int):
    np.random.seed(36631)
    _, axes = plt.subplots(len(labels), n, figsize=(n * 2, len(labels) * 2.2))
    for row, c in enumerate(labels):
        idxs = np.random.choice(np.where(y == c)[0], size=n, replace=False)
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(X[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(f'{LABELS[c]} ({c})', fontsize=16, rotation=0, labelpad=40, va='center')
    plt.suptitle('Class Samples', fontsize=20, y=0.98)
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

def batch_test_plot(results: dict, axes=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for bs, data in results.items():
        if bs == 1:
            label = 'Batch=1 (SGD)'
        elif bs is None:
            label = 'Full-Batch'
        else:
            label = f'Batch={bs}'
        epochs = len(data['val_loss'])
        x = range(1, epochs + 1)
        axes[0].plot(x, data['train_loss'], label=label)
        axes[1].plot(x, data['val_loss'],   label=label)

    for ax, title in zip(axes, ['Train Loss', 'Validation Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy')
        ax.set_title(f'{title} by Batch Size')
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(results),
               framealpha=0.5, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    plt.show()

def optimizer_test_plot(results: dict, axes=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    linestyles = ['-', '--', ':', '-.']
    opt_names = list(dict.fromkeys(label.split('(')[0] for label in results))
    ls_map = {name: linestyles[i % len(linestyles)] for i, name in enumerate(opt_names)}

    for label, data in results.items():
        opt_name = label.split('(')[0]
        epochs = len(data['val_loss'])
        x = range(1, epochs + 1)
        axes[0].plot(x, data['train_loss'], linestyle=ls_map[opt_name], label=label)
        axes[1].plot(x, data['val_loss'],   linestyle=ls_map[opt_name], label=label)

    for ax, title in zip(axes, ['Train Loss', 'Validation Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy')
        ax.set_title(f'{title} by Optimizer')
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(results), 3),
               framealpha=0.5, bbox_to_anchor=(0.5,0))

    plt.tight_layout()
    plt.show()

def lr_scheduling_test_plot(constant_results: dict, linear_results: dict, exponential_results: dict):
    _, axes = plt.subplots(1, 2, figsize=(16, 5))

    constant_label, constant_data = next(iter(constant_results.items()))
    constant_epochs = len(constant_data['val_loss'])

    for ax, results, schedule in zip(axes, [linear_results, exponential_results], ['Linear', 'Exponential']):
        x_const = range(1, constant_epochs + 1)
        ax.plot(x_const, constant_data['val_loss'], label=constant_label, color='black', linestyle='--', linewidth=1.5)
        for label, data in results.items():
            epochs = len(data['val_loss'])
            x = range(1, epochs + 1)
            ax.plot(x, data['val_loss'], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy')
        ax.set_title(f'Validation Loss — {schedule} Scheduling')
        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', framealpha=0.5)

    plt.tight_layout()
    plt.show()

def weight_decay_test_plot(results: dict, axes=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    linestyles = ['-', '--', ':', '-.']
    opt_names = list(dict.fromkeys(label.split('(')[0] for label in results))
    ls_map = {name: linestyles[i % len(linestyles)] for i, name in enumerate(opt_names)}

    for label, data in results.items():
        opt_name = label.split('(')[0]
        epochs = len(data['val_loss'])
        x = range(1, epochs + 1)
        axes[0].plot(x, data['train_loss'], linestyle=ls_map[opt_name], label=label)
        axes[1].plot(x, data['val_loss'],   linestyle=ls_map[opt_name], label=label)

    for ax, title in zip(axes, ['Train Loss', 'Validation Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy')
        ax.set_title(f'{title} by Weight Decay')
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(results), 5),
               framealpha=0.5, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    plt.show()

def training_summary(model: NN, train_hist: dict, X_train, y_train,
                    X_val, y_val, title: str = None):
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

def _has_early_stop(train_hists):
    counts = [len(h['train_loss']) for h in train_hists]
    return min(counts) < max(counts)

def compare_loss_curves(train_hists: list[dict], names: list[str], ax_main=None, ax_zoom=None):
    epoch_counts = [len(h['train_loss']) for h in train_hists]
    max_epochs   = max(epoch_counts)
    early_stopped = [n for n in epoch_counts if n < max_epochs]
    has_zoom = bool(early_stopped)

    standalone = ax_main is None
    if standalone:
        n_cols = 2 if has_zoom else 1
        ratios = [2, 1] if has_zoom else [1]
        _, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols // 1, 5),
                               gridspec_kw={'width_ratios': ratios})
        ax_main = axes[0] if has_zoom else axes
        ax_zoom = axes[1] if has_zoom else None

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (hist, name) in enumerate(zip(train_hists, names)):
        c = colors[i % len(colors)]
        n = len(hist['train_loss'])
        x = range(1, n + 1)
        ax_main.plot(x, hist['train_loss'], color=c, linestyle='--', alpha=0.5)
        ax_main.plot(x, hist['val_loss'],   color=c, linestyle='-')
        ax_main.plot([], [], color=c, linestyle='--', alpha=0.6, label=f'{name} (dev)')
        ax_main.plot([], [], color=c, linestyle='-',             label=f'{name} (test)')
        if ax_zoom is not None:
            ax_zoom.plot(x, hist['train_loss'], color=c, linestyle='--', alpha=0.5)
            ax_zoom.plot(x, hist['val_loss'],   color=c, linestyle='-')

    ax_main.set_xlabel('Epoch')
    ax_main.set_ylabel('Cross Entropy')
    ax_main.set_title('Loss Curves — Final Models')
    ax_main.legend(ncols=2)
    ax_main.grid()

    if ax_zoom is not None and early_stopped:
        zoom_end = max(early_stopped)
        ax_zoom.set_xlim(1, zoom_end)
        all_vals = [v for h in train_hists
                    for v in list(h['train_loss'][:zoom_end]) + list(h['val_loss'][:zoom_end])]
        margin = (max(all_vals) - min(all_vals)) * 0.1 or 0.01
        ax_zoom.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
        for n in early_stopped:
            ax_zoom.axvline(n, color='gray', linestyle=':', linewidth=1)
        ax_zoom.set_xlabel('Epoch')
        ax_zoom.set_title('Early Stop — Zoom')
        ax_zoom.grid()

    if standalone:
        plt.tight_layout()
        plt.show()

def compare_final_metrics(models: list, X, y, names: list[str], n_classes: int, ax: plt.Axes = None):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 5))

    from src.metrics import accuracy, cross_entropy, f1_macro
    from src.utils import to_onehot
    y_oh = to_onehot(y, n_classes)

    metric_names = ['Accuracy', 'F1 Macro', 'Cross Entropy']
    scores = []
    for model in models:
        yhat = model.forward(X)
        scores.append([
            accuracy(yhat, y),
            f1_macro(yhat, y, n_classes),
            cross_entropy(yhat, y_oh),
        ])

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    for (name, s, offset) in zip(names, scores, offsets):
        bars = ax.bar(x + offset, s, width, label=name)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, max(s[i] for s in scores for i in range(len(metric_names))) * 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Metrics')
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    if standalone:
        plt.tight_layout()
        plt.show()

def compare_final_models(models: list, train_hists: list[dict], names: list[str],
                         X, y, n_classes: int = 49, title: str = None):
    has_zoom = _has_early_stop(train_hists)
    n_loss_cols = 2 if has_zoom else 1
    loss_ratios = [2, 1] if has_zoom else [1]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.35)

    top_gs = gridspec.GridSpecFromSubplotSpec(1, n_loss_cols, subplot_spec=gs[0],
                                             width_ratios=loss_ratios, wspace=0.25)
    ax_main = fig.add_subplot(top_gs[0])
    ax_zoom = fig.add_subplot(top_gs[1]) if has_zoom else None

    ax_metrics = fig.add_subplot(gs[1])

    compare_loss_curves(train_hists, names, ax_main=ax_main, ax_zoom=ax_zoom)
    compare_final_metrics(models, X, y, names, n_classes, ax=ax_metrics)

    fig.suptitle(title or 'Final Models Comparison', fontsize=16)
    plt.show()