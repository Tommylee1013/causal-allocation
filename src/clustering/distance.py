import numpy as np
import pandas as pd
import itertools

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed
from tqdm import tqdm

def calculate_mutual_info_pair(
        data : pd.DataFrame,
        asset_i : str,
        asset_j : str
    ) -> tuple :
    "calculate mutual information between asset i and asset j"
    x = data.iloc[:, asset_i].values.reshape(-1, 1)
    y = data.iloc[:, asset_j].values

    mutual_information = mutual_info_regression(
        x, y,
        n_neighbors = 3,
        random_state = 42
    )[0]
    return asset_i, asset_j, mutual_information

def get_mutual_info_matrix(
        data : pd.DataFrame,
        n_jobs : int = -1
    ) -> pd.DataFrame :
    n_assets = data.shape[1]
    asset_names = data.columns

    print(f"Calculating MI for {n_assets} assets ({n_assets*(n_assets-1)//2} pairs)...")

    pairs = list(itertools.combinations(range(n_assets), 2))

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_mutual_info_pair)(data, i, j) for i, j in pairs
    )

    mi_matrix = np.zeros((n_assets, n_assets))

    for i, j, mi in results:
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

    for i in range(n_assets):
        mi_matrix[i, i] = mutual_info_regression(
            data.iloc[:, i].values.reshape(-1, 1),
            data.iloc[:, i].values,
            n_neighbors=3, random_state=42
        )[0]

    dist_matrix = np.zeros((n_assets, n_assets))
    for i, j in itertools.combinations(range(n_assets), 2):
        h_x = mi_matrix[i, i]
        h_y = mi_matrix[j, j]
        mi_xy = mi_matrix[i, j]

        nmi = mi_xy / max(h_x, h_y, 1e-10)
        distance = max(0, 1 - nmi)

        dist_matrix[i, j] = distance
        dist_matrix[j, i] = distance

    return pd.DataFrame(dist_matrix, index=asset_names, columns=asset_names)

def calculate_discrete_mi_matrix(df, bins=10):
    n_assets = df.shape[1]

    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    df_discrete = pd.DataFrame(kbd.fit_transform(df), columns=df.columns)

    h_list = []
    for i in range(n_assets):
        h = mutual_info_score(df_discrete.iloc[:, i], df_discrete.iloc[:, i])
        h_list.append(h)

    dist_matrix = np.zeros((n_assets, n_assets))
    for i in tqdm(range(n_assets)):
        for j in range(i + 1, n_assets):
            mi = mutual_info_score(df_discrete.iloc[:, i], df_discrete.iloc[:, j])

            nmi = mi / max(h_list[i], h_list[j])

            distance = max(0, 1 - nmi)

            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return pd.DataFrame(dist_matrix, index=df.columns, columns=df.columns)

