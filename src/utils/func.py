import pandas as pd
import numpy as np

def base_name(node: str) -> str:
    # 'USOIL@t-0' -> 'USOIL'
    return str(node).split("@", 1)[0]

def get_assets_by_cluster(cluster_mapping):
    cluster_dict = {}
    unique_clusters = sorted(cluster_mapping.unique())

    print(f"=== Result of asset clustering(number of groups : {len(unique_clusters)}) ===\n")

    for c in unique_clusters:
        assets = cluster_mapping[cluster_mapping == c].index.tolist()
        cluster_dict[f'Cluster_{c}'] = assets

        print(f"Cluster {c} (number of asset : {len(assets)}):")
        print(f"   {', '.join(assets)}")
        print("-" * 50)

    return cluster_dict

def zscore(
        s : pd.Series,
        span : int = 252
    ) -> pd.Series :
    mu = s.ewm(span=span, adjust=False).mean()
    sd = s.ewm(span=span, adjust=False).std().replace(0, np.nan)
    return (s - mu) / sd

def fix_psd_cov(
        cov: pd.DataFrame,
        eps: float = 1e-8
    ) -> pd.DataFrame:
    """대칭화 + 고유값 바닥으로 PSD 보정."""
    C = 0.5 * (cov.values + cov.values.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    C_psd = (vecs * vals) @ vecs.T
    return pd.DataFrame(C_psd, index=cov.index, columns=cov.columns)

def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    s = np.sqrt(np.diag(cov.values))
    denom = np.outer(s, s)
    corr = cov.values / np.maximum(denom, 1e-12)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def clean_cov(
        C: pd.DataFrame,
        eps: float = 1e-8
    ) -> pd.DataFrame | None:
    if C is None:
        return None
    C = C.replace([np.inf, -np.inf], np.nan)
    if C.isna().all().all():
        return None
    C = C.fillna(0.0)
    C.values[np.diag_indices_from(C.values)] += eps
    return C

def safe_equal(a_list):
    return pd.Series(1.0 / len(a_list), index=a_list)