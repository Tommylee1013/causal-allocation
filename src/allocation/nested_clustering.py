import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.clustering.distance import dist_from_corr, single_linkage_from_dist, quasi_diag
from src.utils.func import fix_psd_cov, corr_from_cov

def ivp_weights(cov: pd.DataFrame) -> pd.Series:
    """Inverse-Variance Portfolio weights."""
    v = np.diag(cov.values)
    w = 1.0 / np.maximum(v, 1e-12)
    w = w / w.sum()
    return pd.Series(w, index=cov.index)

def nco_weights(
        cov: pd.DataFrame,
        n_clusters: int | None = None
    ) -> pd.Series:
    """
    단순 NCO: (1) intra: 각 클러스터 내 IVP
             (2) inter: 클러스터 대표 포트폴리오들 간 IVP
             (3) 결합: w_i = w_inter[c(i)] * w_intra[i|c]
    """
    cov = fix_psd_cov(cov)
    n = cov.shape[0]
    if n <= 2:
        return ivp_weights(cov)

    if n_clusters is None:
        n_clusters = int(np.clip(np.sqrt(n), 2, min(10, n)))

    corr = corr_from_cov(cov)
    dist = dist_from_corr(corr).values
    linkage = single_linkage_from_dist(dist)

    # quasi-order 후, 등분할로 클러스터 생성(단순)
    order = quasi_diag(linkage)
    ordered_assets = cov.index[order].tolist()
    splits = np.array_split(np.array(ordered_assets), n_clusters)
    clusters = [list(x) for x in splits if len(x) > 0]
    kC = len(clusters)

    # 1) intra: cluster 내 IVP
    w_intra_by_cluster: list[pd.Series] = []
    for cl in clusters:
        sub = cov.loc[cl, cl]
        w_intra_by_cluster.append(ivp_weights(sub).reindex(cl))

    # 2) cluster 대표 포트폴리오 행렬 Wmat 구성
    #    각 클러스터 대표 r_c = sum_i w_intra[i|c] * r_i
    Wmat = np.zeros((n, kC), dtype=float)
    idx_map = {a: i for i, a in enumerate(cov.index)}
    for k, (cl, w_intra_cl) in enumerate(zip(clusters, w_intra_by_cluster)):
        for a, wa in w_intra_cl.items():
            Wmat[idx_map[a], k] = float(wa)

    cov_c = Wmat.T @ cov.values @ Wmat
    cov_c = fix_psd_cov(pd.DataFrame(cov_c, index=range(kC), columns=range(kC)))

    # inter: cluster 간 IVP
    w_inter = ivp_weights(cov_c)  # index: 0..kC-1

    # 3) 결합: w_i = w_inter[k] * w_intra[i|k]
    w = pd.Series(0.0, index=cov.index, dtype=float)
    for k, (cl, w_intra_cl) in enumerate(zip(clusters, w_intra_by_cluster)):
        w.loc[cl] = float(w_inter.loc[k]) * w_intra_cl.values

    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("zero-sum weights")
    w = w / s
    return w

def auto_select_k_from_cov_with_existing_funcs(
    cov: pd.DataFrame,
    k_min: int = 2,
    k_max: int | None = None,
    linkage_method: str = "average",
) -> int:
    """
    기존에 정의된:
      - _corr_from_cov(cov) -> corr
      - _dist_from_corr(corr) -> distance (Prado distance)
    를 그대로 사용해서 NCO 내부 클러스터 수 k를 자동 선택.
    """

    n = cov.shape[0]
    if n < 3:
        return 1

    if k_max is None:
        k_max = min(10, n - 1)
    k_min = max(2, k_min)
    k_max = max(k_min, min(k_max, n - 1))

    corr = corr_from_cov(cov)
    dist = dist_from_corr(corr).values

    # linkage는 condensed distance 필요
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method=linkage_method)

    best_k = k_min
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")

        if len(np.unique(labels)) < 2:
            continue

        try:
            score = silhouette_score(dist, labels, metric="precomputed")
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_k = k

    return best_k