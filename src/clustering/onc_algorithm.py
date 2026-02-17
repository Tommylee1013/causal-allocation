import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tqdm import tqdm

def evaluate_k(
        k : int,
        linkage_matrix : np.ndarray,
        dist_matrix : np.ndarray,
    ) -> tuple:
    """
    Evaluates the clustering quality for a given k using the silhouette score.
    """
    # Assign cluster labels based on the linkage matrix for a specific k
    labels = fcluster(linkage_matrix, t=k, criterion='maxclust')

    # Handle cases where clustering fails to produce at least 2 clusters
    if len(np.unique(labels)) < 2:
        return k, -1

    # Calculate silhouette score using the precomputed distance matrix
    score = silhouette_score(
        dist_matrix,
        labels,
        metric='precomputed'
    )
    return k, score

def get_cluster_representatives(
    data: pd.DataFrame,
    mi_dist_df: pd.DataFrame,
    max_k: int = 10,
    n_jobs: int = -1
) -> tuple:
    """
    Performs Optimal Number of Clusters (ONC) search and extracts the
    first principal component (PC1) for each cluster.

    Args:
        data: Standardized/FracDiff time-series DataFrame (Assets as columns).
        mi_dist_df: Precomputed Mutual Information distance matrix.
        max_k: Maximum number of clusters to explore.
        n_jobs: Number of CPU cores for parallel processing (-1 uses all).

    Returns:
        pc1_df: DataFrame containing the PC1 time-series for each cluster.
        clusters: Series mapping each asset to its assigned cluster.
    """
    print(f"🚀 Starting ONC Algorithm: Parallel search from k=2 to {max_k} (n_jobs={n_jobs})")

    # Perform hierarchical clustering using the average linkage method
    dist_array = mi_dist_df.values
    linkage_matrix = linkage(dist_array, method='average')

    # Parallel search for the optimal k based on the silhouette score
    # Wrapped in list brackets [] to avoid SyntaxError in certain Python environments
    results = Parallel(n_jobs=n_jobs)(
        [delayed(evaluate_k)(k, linkage_matrix, dist_array) for k in tqdm(range(2, max_k + 1), desc="Progress of ONC Exploration")]
    )

    # Extract the k value with the highest silhouette score
    best_k, best_score = max(results, key=lambda x: x[1])
    print(f"Optimal number of clusters found: {best_k} (Best Silhouette Score: {best_score:.4f})")

    # Perform final clustering with the optimal k
    final_labels = fcluster(
        linkage_matrix,
        t = best_k,
        criterion = 'maxclust'
    )
    clusters = pd.Series(final_labels, index=data.columns)

    # Extract the First Principal Component (PC1) for each cluster node
    pc1_dict = {}
    for c in range(1, best_k + 1):
        assets_in_cluster = clusters[clusters == c].index
        cluster_data = data[assets_in_cluster]

        # Condense the cluster information into a single latent causal node via PCA
        pca = PCA(n_components=1)
        pc1_series = pca.fit_transform(cluster_data).flatten()
        pc1_dict[f'Cluster_{c}'] = pc1_series

    # Construct the final DataFrame for Causal Discovery (NOTEARS/DYNOTEARS)
    pc1_df = pd.DataFrame(pc1_dict, index=data.index)

    return (pc1_df, clusters)