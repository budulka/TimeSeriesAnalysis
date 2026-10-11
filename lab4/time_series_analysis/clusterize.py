import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def prepare_data(model):
    return np.asarray(model).reshape(-1, 1)


def compute_inertias(data, k_range):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(data)
        inertias.append(km.inertia_)
    return inertias


def find_best_k(k_range, inertias):
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    best_idx = np.argmax(diffs2) + 1
    return list(k_range)[best_idx]


def plot_elbow(model):
    data = prepare_data(model)
    k_range = range(1, 11)
    inertias = compute_inertias(data, k_range)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()


def plot_clusters(data, labels, centers):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(data)), data, c=labels, cmap='viridis', s=10)
    for c in centers:
        plt.axhline(c, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('KMeans Clustering')
    plt.show()


def run_clustering(model, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    data = prepare_data(model)
    labels = km.fit_predict(data)
    plot_clusters(data, labels, km.cluster_centers_.flatten())
 


