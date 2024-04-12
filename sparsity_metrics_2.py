import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import umap
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data with different sparsity levels
def generate_sparse_data(num_samples, input_dim, sparsity_level):
    data = make_blobs(n_samples=num_samples, n_features=input_dim, centers=2, cluster_std=1.0)[0]
    mask = np.random.rand(num_samples, input_dim) < sparsity_level
    data[mask] = 0  # Apply sparsity mask
    return data

# Compute AIC and BIC for a Gaussian Mixture Model
def compute_information_criteria(X, num_components):
    gmm = GaussianMixture(n_components=num_components, random_state=42).fit(X)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    return aic, bic

# Compute explained variance ratio for PCA
def compute_explained_variance_ratio(X, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    return explained_variance_ratio

# Compute UMAP embeddings and metrics for dimensionality reduction quality
def compute_umap_metrics(X, num_neighbors):
    umap_model = umap.UMAP(n_neighbors=num_neighbors, random_state=42)
    embedding = umap_model.fit_transform(X)
    
    # Compute distances in original and UMAP-embedded spaces
    original_distances = np.sum(np.abs(X - np.mean(X, axis=0)), axis=1)
    umap_distances = np.sum(np.abs(embedding - np.mean(embedding, axis=0)), axis=1)
    
    # Calculate trustworthiness and continuity
    trustworthiness = 1 - mean_squared_error(original_distances, umap_distances) / np.var(original_distances)
    continuity = 1 - mean_squared_error(original_distances, umap_distances) / np.var(umap_distances)
    
    return trustworthiness, continuity

# Define parameters
input_samples = 1000
input_dim = 20  # Input dimension
sparsity_levels = [0.1, 0.3, 0.5, 0.7]  # Sparsity levels to test
num_components = 2  # Number of components for GMM
num_neighbors = 15  # Number of neighbors for UMAP

# Initialize data storage
results = []

for sparsity in sparsity_levels:
    X = generate_sparse_data(input_samples, input_dim, sparsity)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    aic, bic = compute_information_criteria(X_scaled, num_components)
    explained_variance_ratio = compute_explained_variance_ratio(X_scaled, input_dim)
    trustworthiness, continuity = compute_umap_metrics(X_scaled, num_neighbors)

    results.append({
        'Sparsity': sparsity,
        'AIC': aic,
        'BIC': bic,
        'Explained Variance Ratio': np.mean(explained_variance_ratio),
        'Trustworthiness': trustworthiness,
        'Continuity': continuity
    })

# Convert results to DataFrame for easier analysis and visualization
results_df = pd.DataFrame(results)
print(results_df)

# Plotting results
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(results_df['Sparsity'], results_df['AIC'], marker='o')
plt.title('AIC vs. Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('AIC')

plt.subplot(2, 2, 2)
plt.plot(results_df['Sparsity'], results_df['BIC'], marker='o')
plt.title('BIC vs. Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('BIC')

plt.subplot(2, 2, 3)
plt.plot(results_df['Sparsity'], results_df['Explained Variance Ratio'], marker='o')
plt.title('Explained Variance Ratio vs. Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('Explained Variance Ratio')

plt.subplot(2, 2, 4)
plt.plot(results_df['Sparsity'], results_df['Trustworthiness'], marker='o', label='Trustworthiness')
plt.plot(results_df['Sparsity'], results_df['Continuity'], marker='o', label='Continuity')
plt.title('UMAP Metrics vs. Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('Metric Value')
plt.legend()

plt.tight_layout()
plt.show()
