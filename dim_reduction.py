import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.datasets import make_blobs
import os
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress deprecation warnings and specific user warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs value 1 overridden.*")

def generate_sample_data(n_samples=500):
    """Generate synthetic clustered data"""
    X, y = make_blobs(n_samples=n_samples, centers=4, n_features=10, random_state=42)
    return X, y

def reduce_dimensions(X, method='pca', n_components=2):
    """Apply dimensionality reduction"""
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
    
    return reducer.fit_transform(X)

def plot_results(X_reduced, y, method='PCA', dim=2):
    """Visualize reduced dimensions"""
    plt.figure(figsize=(10, 8))
    
    if dim == 2:
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Spectral', alpha=0.7)
    elif dim == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='Spectral', alpha=0.7)
        ax.set_zlabel('Component 3')
    
    plt.title(f'{method} Projection ({dim}D)', fontsize=16, pad=20)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    
    # Save plot
    os.makedirs('assets', exist_ok=True)
    plt.savefig(f'assets/{method.lower()}_plot.png', dpi=120, bbox_inches='tight')
    plt.show()

def main():
    # Generate or load data
    X, y = generate_sample_data()
    
    # Reduce dimensions with different methods
    methods = ['pca', 'tsne', 'umap']
    dimensions = [2, 3]  # Try both 2D and 3D
    
    for method in methods:
        for dim in dimensions:
            print(f"Processing {method.upper()} ({dim}D)...")
            X_reduced = reduce_dimensions(X, method, dim)
            plot_results(X_reduced, y, method.upper(), dim)

if __name__ == "__main__":
    main()
