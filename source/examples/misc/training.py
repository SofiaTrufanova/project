import torch
import torchfd
import numpy as np
import pandas as pd

from misc import infomax

from collections import defaultdict

from tqdm import tqdm, trange
from IPython.display import clear_output

# Saving history
import pandas as pd
import json
import os
import pathlib
from datetime import datetime

# Plotting.
import matplotlib
from matplotlib import pyplot as plt

# Clustering.
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Classification.
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Statistical tests for distributions
import scipy.stats as sps
import pingouin

from .plots import plot_history, plot_embeddings
from .statistics import *


def convert_to_embeddings(embedder, dataloader, device):
    was_in_trainig = embedder.training
    embedder.eval()
    
    # Targets and predictions.
    y_all = []
    embeddings_all = []
    
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch

            y_all.append(y.detach().cpu())
            embeddings_all.append(embedder(x.to(device)).detach().cpu())
        
    embeddings_all = torch.vstack(embeddings_all)
    y_all = torch.concat(y_all)

    embedder.train(was_in_trainig)

    return embeddings_all, y_all


def train_infomax_embedder(
    infomax_embedder, train_dataloader, test_dataloader, device,
    n_epochs=500,
    loss=torchfd.loss.InfoNCELoss(),
    optimizer_embedder_network=lambda params: torch.optim.Adam(params, lr=1.0e-3),
    optimizer_discriminator_network=lambda params: torch.optim.Adam(params, lr=1.0e-3),
    callback: callable=None,
    distribution: str="normal"
):
    
    history = defaultdict(lambda: defaultdict(list))

    optimizer_embedder_network = optimizer_embedder_network(infomax_embedder.embedder_network.parameters())
    optimizer_discriminator_network = optimizer_discriminator_network(infomax_embedder.discriminator_network.parameters())

    step = 0
    for epoch in trange(n_epochs):        
        mean_loss = 0.0

        n_batches = 0
        for batch in train_dataloader:
            optimizer_embedder_network.zero_grad()
            optimizer_discriminator_network.zero_grad()
            
            x, y = batch
            x = x.to(device)
            
            _loss = loss(*infomax_embedder(x))
            _loss.backward()

            optimizer_embedder_network.step()
            optimizer_discriminator_network.step()
            
            mean_loss += _loss.item()
            n_batches += 1

        step += n_batches
        mean_loss /= n_batches

        history["training"]["loss"].append((epoch, step, mean_loss))
        history["training"]["mutual_information"].append((epoch, step, -mean_loss))
        history["training"]["kullback_leibler_upper_bound"].append(
            (epoch, step, infomax_embedder.embedder_network.embedding_dim * infomax_embedder.output_channel.capacity + mean_loss)
        )

        #scheduler_embedder_network.step()
        #scheduler_discriminator_network.step()

        clear_output(wait=True)

        # Callback
        if not (callback is None):
            callback(history, epoch, step, infomax_embedder, train_dataloader, test_dataloader, device)

        # Plots.
        for name, subhistory in history.items():
            plot_history(subhistory, title=name)

        with torch.no_grad():
            infomax_embedder.embedder_network.eval()
            
            x_lim = y_lim = (-3.0, 3.0) if distribution == "normal" else (-0.1, 1.1)
            x, y = next(iter(train_dataloader))
            plot_embeddings(infomax_embedder.embedder_network(x.to(device)).detach().cpu().numpy(), y.detach().cpu().numpy(), x_lim=x_lim, y_lim=y_lim)
                
            infomax_embedder.embedder_network.train()

        plt.show();
        
    return history


def classification_callback(history, epoch, step, infomax_embedder, train_dataloader, test_dataloader, device,
                            period=50,
                            distribution_tests={
                                "henze_zirkler": BootstrappedMultivariateNormalTest(),
                                "shapiro_wilk": BootstrappedRandomProjectionUnivariateNormalTest(),
                                "dagostino_pearson": BootstrappedRandomProjectionUnivariateNormalTest(test=lambda X: sps.normaltest(X, axis=0).pvalue),
                            },
                            clustering_metrics={
                                "silhouette_score": silhouette_score,
                                "davies_bouldin_score": davies_bouldin_score,
                                "calinski_harabasz_score": calinski_harabasz_score,
                            },
                            classifiers={
                                #"logistic_regression": lambda: SGDClassifier(loss='log_loss'),
                                "logistic_regression": LogisticRegression,
                                "gaussian_naive_bayes": GaussianNB,
                                "knn": KNeighborsClassifier,
                                "mlp": lambda: MLPClassifier(alpha=1.0, max_iter=1000),
                            },
                            classification_metrics={
                                "roc_auc":  lambda y_true, y_pred, proba_pred: roc_auc_score(y_true, proba_pred, multi_class='ovo'),
                                "accuracy": lambda y_true, y_pred, proba_pred: accuracy_score(y_true, y_pred),
                                "f1":       lambda y_true, y_pred, proba_pred: f1_score(y_true, y_pred, average='micro'),
                            }):
    if epoch % period:
        return

    train_embeddings, train_y = convert_to_embeddings(infomax_embedder.embedder_network, train_dataloader, device)
    train_embeddings = train_embeddings.detach().cpu().numpy()
    train_y = train_y.detach().cpu().numpy()

    test_embeddings, test_y = convert_to_embeddings(infomax_embedder.embedder_network, test_dataloader, device)
    test_embeddings = test_embeddings.detach().cpu().numpy()
    test_y = test_y.detach().cpu().numpy()

    # Distribution tests.
    if distribution_tests:
        for test_name, test in distribution_tests.items():
            history["distribution"][f"{test_name}_(train)"].append((epoch, step, test(train_embeddings)))
            history["distribution"][f"{test_name}_(test)"].append((epoch, step, test(test_embeddings)))

    # Clustering.
    if clustering_metrics:
        for metric_name, metric in clustering_metrics.items():
            history["clustering"][metric_name].append((epoch, step, metric(test_embeddings, test_y)))

    # Classification.
    if classifiers and classification_metrics:
        for classifier_name, factory in classifiers.items():
            classifier = factory()
            classifier.fit(train_embeddings, train_y)
    
            test_proba_pred = classifier.predict_proba(test_embeddings)
            test_y_pred = np.argmax(test_proba_pred, axis=-1)
            
            for metric_name, metric in classification_metrics.items():
                history[f"classification_{classifier_name}"][metric_name].append((epoch, step, metric(test_y, test_y_pred, test_proba_pred)))


def save_results(model, config, history, path, averaging_epochs: int=101):
    folder_name = f"{config['distribution']}_{config['embedding_dim']}_{config['output_p']:.1e}_{datetime.now().strftime('%d-%b-%Y_%H:%M:%S')}"
    folder_path = path / folder_name
    
    os.makedirs(folder_path, exist_ok=True)

    averaged_values = defaultdict(dict)
    for category_name, category_dict in history.items():
        #category_path = folder_path / category_name

        dataframe = pd.DataFrame(columns=["epoch", "step"])
        for subcategory_name, sebcategory_list in category_dict.items():
            subcategory_dataframe = pd.DataFrame(sebcategory_list, columns=["epoch", "step", subcategory_name])
            dataframe = dataframe.merge(subcategory_dataframe, how="outer")

            selected_values = subcategory_dataframe[
                subcategory_dataframe["epoch"] > subcategory_dataframe["epoch"].max() - averaging_epochs
            ][subcategory_name]
            
            averaged_values[category_name][subcategory_name] = float(selected_values.mean())
            averaged_values[category_name][subcategory_name + "_std"] = float(selected_values.std())
        
        dataframe.to_csv(folder_path / f"{category_name}.csv", index=False)
    
    with open(folder_path / "config.json", 'w') as file:
        json.dump(config, file, indent=4)

    with open(folder_path / "averaged_values.json", 'w') as file:
        json.dump(averaged_values, file, indent=4)

    torch.save(model, folder_path / "model.pt")
