import numpy as np

import matplotlib
from matplotlib import pyplot as plt


def plot_history(history, n_columns=4, figsize=(4, 3), title=None):
    n_plots = len(history)
    n_columns = min(n_columns, n_plots)
    n_rows = int(np.ceil(n_plots / n_columns))
    
    figure, axes = plt.subplots(n_rows, n_columns)
    if not (title is None):
        figure.suptitle(title)

    figure.set_figwidth(figsize[0] * n_columns)
    figure.set_figheight(figsize[1] * n_rows)

    for index, (name, data) in enumerate(sorted(history.items())):
        axes.flat[index].set_title(name)
        axes.flat[index].plot([x[1] for x in data], [x[2] for x in data])
        axes.flat[index].grid()

    plt.show();

    
def plot_estimated_MI_trainig(true_mi: float, epochs, estimated_MI, estimated_latent_MI=None):
    """
    Plot mutual information estimate during training.
    
    Parameters
    ----------
    true_mi : float
        True value of the mutual information
    epochs : iterable
        Epochs array (x axis)
    estimated_MI : iterable
        Mutual iformation estimates during training
    estimated_latent_MI : iterable (optional)
        Mutual iformation estimates based on latent representation during training
    """
    
    fig, ax = plt.subplots()

    fig.set_figheight(9)
    fig.set_figwidth(16)

    # Grid.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title("Mutual information estimate while training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$ I(X,Y) $")
    
    ax.minorticks_on()

    if not estimated_latent_MI is None:
        ax.plot(epochs, estimated_latent_MI, label="$ \\hat I(\\xi,\\eta) $")
    ax.plot(epochs, estimated_MI, label="$ \\hat I(X,Y) $")
    ax.hlines(y=true_mi, xmin=min(epochs), xmax=max(epochs), color='red', label="$ I(X,Y) $")

    ax.legend(loc='upper left')

    plt.show();


def plot_embeddings(embeddings: np.array, labels: np.array,
                    alpha: float=0.5, size: float=40.0, legend_size=50.0,
                    x_lim=(-3.0, 3.0), y_lim=(-3.0, 3.0)):
    """
    Plot embeddings projection.
    
    Parameters
    ----------
    embeddings_tensor : np.array
        Array of embeddigns
    labels : np.array
        Array of labels
    """
    
    fig, ax = plt.subplots()

    fig.set_figheight(10)
    fig.set_figwidth(10)

    # Grid.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title("Embedding space plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    delta = 1.0
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    
    ax.minorticks_on()

    classes = np.unique(labels)
    for label in classes:
        selected = embeddings[labels==label]
        ax.scatter(selected[:,0], selected[:,1], label=str(label), alpha=alpha, s=size)

    legend = ax.legend(loc='lower right')

    for legend_handle in legend.legend_handles:
        legend_handle.set_sizes([legend_size])

    plt.show();