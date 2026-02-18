import numpy as np
import pandas as pd

from src.clustering.distance import quasi_diag, single_linkage_from_dist, dist_from_corr
from src.utils.func import fix_psd_cov, corr_from_cov
from src.allocation.nested_clustering import  ivp_weights

def hrp_cluster_variance(
        C_ordered: pd.DataFrame,
        items: list[str]
    ) -> float:
    """
    HRP에서 클러스터(서브셋) 분산 계산.
    기본 구현: IVP로 클러스터 대표 포트폴리오를 만든 뒤 v = w^T Σ w.
    """
    sub = C_ordered.loc[items, items]
    w_ivp = ivp_weights(sub)  # index=items
    wv = w_ivp.to_numpy(dtype=float).reshape(-1, 1)
    return float(wv.T @ sub.to_numpy(dtype=float) @ wv)


def hrp_recursive_bisect(
        C_ordered: pd.DataFrame,
        items: list[str],
        w: pd.Series,
        eps: float = 1e-12,
    ) -> None:
    """
    HRP 재귀 분할로 w를 in-place로 업데이트.
    C_ordered: quasi-diag 순서로 정렬된 공분산 (index/columns=ordered_assets)
    items: 현재 클러스터에 속한 ordered_assets의 부분 리스트
    w: ordered_assets index를 가진 Series (in-place 업데이트)
    """
    n = len(items)
    if n <= 1:
        return

    k = n // 2
    left = items[:k]
    right = items[k:]

    vL = hrp_cluster_variance(C_ordered, left)
    vR = hrp_cluster_variance(C_ordered, right)

    alpha = 1.0 - vL / (vL + vR + eps)  # Prado 방식
    w.loc[left] *= alpha
    w.loc[right] *= (1.0 - alpha)

    hrp_recursive_bisect(C_ordered, left, w, eps=eps)
    hrp_recursive_bisect(C_ordered, right, w, eps=eps)


def hrp_ordered_assets_from_cov(cov: pd.DataFrame) -> list[str]:
    """
    cov -> corr -> dist -> linkage -> quasi-diag 순서로 HRP 자산 정렬(ordered_assets) 생성.
    """
    cov = fix_psd_cov(cov)
    corr = corr_from_cov(cov)
    dist = dist_from_corr(corr).to_numpy(dtype=float)

    linkage = single_linkage_from_dist(dist)
    order = quasi_diag(linkage)  # list[int]
    return cov.index[order].tolist()


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """
    HRP 기본 구현(단순화):
    - cluster variance: IVP 기반
    - linkage: single
    - split: ordered list를 반으로 계속 쪼개는 재귀(bisection)
    """
    cov = fix_psd_cov(cov)
    ordered_assets = hrp_ordered_assets_from_cov(cov)

    C_ordered = cov.loc[ordered_assets, ordered_assets]
    w_ordered = pd.Series(1.0, index=ordered_assets, dtype=float)

    hrp_recursive_bisect(C_ordered, ordered_assets, w_ordered, eps=1e-12)

    w_ordered = w_ordered / float(w_ordered.sum())
    return w_ordered.reindex(cov.index).fillna(0.0)