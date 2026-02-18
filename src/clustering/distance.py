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

def dist_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    # Prado distance
    d = np.sqrt(0.5 * (1.0 - corr))
    return d

def quasi_diag(linkage: np.ndarray) -> list[int]:
    """HRP quasi-diagonalization: SciPy 없이 linkage 행렬을 직접 다룸.
    linkage: (n-1) x 4 (i, j, dist, count)
    """
    linkage = linkage.astype(int)
    n = linkage.shape[0] + 1
    # 마지막 merge의 cluster id는 n + (n-2)
    sort_ix = [n + linkage.shape[0] - 1]

    def _children(k):
        if k < n:
            return [k]
        i, j = linkage[k - n, 0], linkage[k - n, 1]
        return _children(i) + _children(j)

    return _children(sort_ix[0])

def single_linkage_from_dist(D: np.ndarray) -> np.ndarray:
    """linkage 유사 구조.
    (정교한 clustering이 아니라 HRP ordering용으로만 사용)
    """
    n = D.shape[0]
    # Kruskal 스타일로 MST 만든 뒤 merge 기록(단순화)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i, j], i, j))
    edges.sort(key=lambda x: x[0])

    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False, ra
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True, ra

    # linkage rows: [i, j, dist, new_count]
    linkage = []
    next_cluster_id = n
    # 클러스터 대표 id를 유지하기 위해 map
    rep = {i: i for i in range(n)}
    cnt = {i: 1 for i in range(n)}

    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # merge
        ci, cj = rep[ri], rep[rj]
        new_count = cnt[ri] + cnt[rj]
        linkage.append([ci, cj, float(dist), float(new_count)])

        ok, rnew = union(ri, rj)
        # 새 클러스터 id 할당
        rep[rnew] = next_cluster_id
        cnt[rnew] = new_count
        next_cluster_id += 1

        if len(linkage) == n - 1:
            break

    return np.array(linkage, dtype=float)