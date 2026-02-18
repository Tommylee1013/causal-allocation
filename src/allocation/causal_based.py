import numpy as np
import pandas as pd
import logging
import cvxpy as cp
import networkx as nx

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.allocation.nested_clustering import nco_weights, auto_select_k_from_cov_with_existing_funcs
from src.allocation.hierarchical_risk_parity import hrp_weights
from src.utils.func import zscore

def cluster_signal_from_parents(
        df: pd.DataFrame,
        G_pruned: nx.DiGraph,
        cluster_vars: list[str],
        macro_z_span: int = 252,
        clip_z: float = 3.0,
        normalize_edge_weights: bool = True,
    ) -> pd.DataFrame:
    """
    pruned DAG의 in-edges(부모)와 weight를 이용해 각 Cluster의 일별 score(신호) 생성.
    - 입력 df: 매크로+클러스터 컬럼이 모두 있는 feature matrix (인덱스=DatetimeIndex)
    - 매크로 변수는 z-score(ewm)로 정규화해서 scale 문제를 줄임
    - 부모가 클러스터인 경우(Cluster->Cluster)는 그 부모 신호를 사용(재귀)해야 하므로,
      위상정렬(topological order) 기준으로 순차 계산.
    """
    X = df.copy()

    # 매크로 후보: 클러스터 제외한 나머지 중 그래프에 존재하는 것
    macro_like = [c for c in X.columns if (c in G_pruned.nodes) and (c not in cluster_vars)]

    # 매크로는 z-score로 스케일 정리
    Z = pd.DataFrame(index=X.index)
    for c in macro_like:
        z = zscore(X[c].astype(float), span=macro_z_span).clip(-clip_z, clip_z)
        Z[c] = z

    # 클러스터 score 저장
    S = pd.DataFrame(index=X.index, columns=cluster_vars, dtype=float)

    # DAG라고 가정(아니면 아래에서 실패 가능)
    order = list(nx.topological_sort(G_pruned))

    # 클러스터 노드만, 위상정렬 순서대로 계산
    for node in order:
        if node not in cluster_vars:
            continue
        parents = list(G_pruned.predecessors(node))
        if len(parents) == 0:
            S[node] = 0.0
            continue

        weights = []
        parent_series = []
        for p in parents:
            w = float(G_pruned[p][node].get("weight", 1.0))
            weights.append(w)

            if p in cluster_vars:
                # 부모가 클러스터면, 이미 계산된 S[p]를 사용
                parent_series.append(S[p].astype(float))
            else:
                # 부모가 매크로면 z-score된 Z[p] 사용
                if p not in Z.columns:
                    # df에 없거나 그래프/df mismatch면 0 처리
                    parent_series.append(pd.Series(0.0, index=X.index))
                else:
                    parent_series.append(Z[p].astype(float))

        w = np.array(weights, dtype=float)

        if normalize_edge_weights:
            denom = np.sum(np.abs(w))
            if denom > 0:
                w = w / denom

        # score_t = sum_j w_j * parent_j_t
        s = np.zeros(len(X.index), dtype=float)
        for wj, sj in zip(w, parent_series):
            s += wj * sj.to_numpy()

        S[node] = s

    return S

def weights_from_cluster_scores(
        cluster_scores: pd.DataFrame,
        rebalance: str = "M",          # "M" 월말 리밸런싱
        long_only: bool = True,
        cash_node: str | None = None,  # "CASH" 같은 현금 노드 쓰려면 지정
        temperature: float = 1.0,      # softmax 온도
        max_weight: float = 0.80,      # 한 클러스터 최대비중
        min_weight: float = 0.0,       # 롱온리면 0
    ) -> pd.DataFrame :
    """
    Cluster score -> 포트폴리오 비중.
    기본: softmax(score/temperature)로 양의 비중 생성 (롱온리)
    """
    S = cluster_scores.copy().astype(float)

    # 리밸런싱 시점 샘플링: 월말 값 사용
    S_reb = S.resample(rebalance).last()

    if long_only:
        # softmax
        X = (S_reb / max(temperature, 1e-12)).to_numpy()
        X = X - X.max(axis=1, keepdims=True)  # 안정화
        W = np.exp(X)
        W = W / W.sum(axis=1, keepdims=True)
        W = pd.DataFrame(W, index=S_reb.index, columns=S_reb.columns)
    else:
        # 롱숏이면 score를 그대로 사용 후 L1 정규화 (원하면 수정)
        W = S_reb.copy()
        denom = W.abs().sum(axis=1).replace(0, np.nan)
        W = W.div(denom, axis=0).fillna(0.0)

    # 클리핑 + 재정규화
    W = W.clip(lower=min_weight, upper=max_weight)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 현금 노드 추가(선택): (1 - sum(weights))를 현금으로
    if cash_node is not None:
        cash = 1.0 - W.sum(axis=1)
        W[cash_node] = cash.clip(lower=0.0)
        # 다시 정규화
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 일별로 forward-fill해서 보유 비중 시계열로 변환
    W_daily = W.reindex(S.index).ffill().bfill()
    return W_daily

def expand_cluster_weights_to_assets(
    w_cluster: pd.DataFrame,
    cluster_to_assets: dict[str, list[str]],
    asset_weighting: str = "equal",      # "equal" | "hrp" | "nco"
    prices: pd.DataFrame | None = None,
    cov_lookback: int = 252,
    nco_n_clusters: int | str | None = "auto",   # ✅ auto 지원
    nco_k_min: int = 2,                           # ✅ auto 탐색 하한
    nco_k_max: int = 10,                          # ✅ auto 탐색 상한
    eps_cov: float = 1e-8,
    rebalance: str = "ME",
    max_nan_frac: float = 0.3,
    min_assets: int = 5,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    클러스터 비중(w_cluster)을 클러스터 내부 자산으로 분배.
    - equal: 동일가중
    - hrp:  클러스터 내부 HRP
    - nco:  클러스터 내부 NCO (nco_n_clusters="auto"면 날짜별 최적 k 선택)

    필요 함수(이미 있다고 가정):
    - _hrp_weights(cov_df) -> pd.Series(index=assets)
    - _nco_weights(cov_df, n_clusters: int) -> pd.Series(index=assets)
    - _auto_select_k_from_cov_with_existing_funcs(cov_df, k_min, k_max, linkage_method="average") -> int
    """

    asset_weighting = asset_weighting.lower()
    logging.info(f"[INIT] expand_cluster_weights_to_assets | mode={asset_weighting} | rebalance={rebalance}")

    idx = w_cluster.index
    assets = sorted({a for lst in cluster_to_assets.values() for a in lst})
    W = pd.DataFrame(0.0, index=idx, columns=assets)

    if asset_weighting in ("hrp", "nco") and prices is None:
        raise ValueError("asset_weighting='hrp' or 'nco' requires prices DataFrame.")

    if prices is not None:
        prices = prices.reindex(idx).ffill()
        logging.info(f"[INIT] prices aligned: {prices.shape}")
        rets_all = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    else:
        rets_all = None

    # 리밸런싱 날짜만
    rb_dates = w_cluster.resample(rebalance).last().index
    rb_dates = [d for d in rb_dates if d in idx]
    logging.info(f"[INIT] rebalance dates: {len(rb_dates)}")

    def _cov_at(dt, a_list):
        window = rets_all[a_list].loc[:dt].tail(cov_lookback)
        window = window.dropna(how="all")
        if window.shape[0] < 2:
            logging.warning(f"[COV_SKIP] {dt.date()} insufficient rows={window.shape[0]}")
            return None, []

        # NaN 비율 높은 자산 제거
        keep_cols = window.columns[window.isna().mean() <= max_nan_frac]
        window = window[keep_cols]
        if window.shape[1] < max(min_assets, 2):
            logging.warning(f"[COV_SKIP] {dt.date()} insufficient cols={window.shape[1]}")
            return None, []

        # ffill/bfill 후에도 NaN 남으면 컬럼 제거
        window = window.ffill().bfill()
        keep_cols2 = window.columns[~window.isna().any()]
        window = window[keep_cols2]
        a_list2 = list(keep_cols2)

        if len(a_list2) < max(min_assets, 2):
            logging.warning(f"[COV_SKIP] {dt.date()} cols after clean={len(a_list2)}")
            return None, []

        C = window.cov().replace([np.inf, -np.inf], np.nan)
        if C.isna().any().any():
            logging.warning(f"[COV_NAN] {dt.date()} cov has nan | cols={len(a_list2)}")
            return None, []

        C.values[np.diag_indices_from(C.values)] += eps_cov
        return C, a_list2

    step = 0
    total_steps = len(cluster_to_assets) * max(1, len(rb_dates))
    logging.info(f"[INIT] steps(upper): {total_steps}")

    for c, a_list in cluster_to_assets.items():
        if c not in w_cluster.columns or len(a_list) == 0:
            logging.warning(f"[SKIP_CLUSTER] {c}")
            continue

        a_list = [a for a in a_list if a in W.columns]
        if len(a_list) == 0:
            logging.warning(f"[SKIP_CLUSTER_EMPTY] {c}")
            continue

        logging.info(f"[CLUSTER] {c} | assets={len(a_list)}")

        if asset_weighting == "equal":
            w_in = pd.Series(1.0 / len(a_list), index=a_list)
            W[a_list] += w_cluster[c].values.reshape(-1, 1) * w_in.values.reshape(1, -1)
            logging.info(f"[EQUAL_DONE] {c}")
            continue

        # 리밸런싱 날짜에서만 intra 계산
        intra_rb = pd.DataFrame(0.0, index=rb_dates, columns=a_list)

        for dt in rb_dates:
            step += 1
            wc = float(w_cluster.at[dt, c])
            if wc == 0.0:
                continue

            C, a_list2 = _cov_at(dt, a_list)

            if C is None:
                w_eff = pd.Series(1.0 / len(a_list), index=a_list)
                logging.warning(f"[FALLBACK_EQ] {dt.date()} | {c}")
            else:
                try:
                    if asset_weighting == "hrp":
                        w2 = hrp_weights(C)  # index=a_list2
                    else:
                        # ✅✅✅ 여기(=NCO)만 바뀐 핵심: k 자동선택
                        if nco_n_clusters is None or (isinstance(nco_n_clusters, str) and nco_n_clusters.lower() == "auto"):
                            k_opt = auto_select_k_from_cov_with_existing_funcs(
                                C,
                                k_min=nco_k_min,
                                k_max=min(nco_k_max, C.shape[0]),
                                linkage_method="average"
                            )
                        else:
                            k_opt = int(nco_n_clusters)

                        logging.info(f"[NCO_K] {dt.date()} | {c} | k_opt={k_opt}")
                        w2 = nco_weights(C, n_clusters=k_opt)

                    if not isinstance(w2, pd.Series):
                        w2 = pd.Series(w2, index=a_list2)

                    w2 = w2.reindex(a_list2).fillna(0.0)
                    s = float(w2.sum())
                    if (not np.isfinite(s)) or s <= 0:
                        raise ValueError("zero-sum weights")

                    w2 = w2 / s

                    w_eff = pd.Series(0.0, index=a_list)
                    w_eff.loc[a_list2] = w2.values

                except Exception as e:
                    w_eff = pd.Series(1.0 / len(a_list), index=a_list)
                    logging.error(f"[FAIL_{asset_weighting.upper()}] {dt.date()} | {c} | {repr(e)}")

            intra_rb.loc[dt, a_list] = wc * w_eff.values

            if step % log_every == 0:
                logging.info(f"[PROGRESS] {step}/{total_steps}")

        intra_daily = intra_rb.reindex(idx).ffill().fillna(0.0)
        W[a_list] += intra_daily[a_list]

    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    logging.info("[DONE] expand_cluster_weights_to_assets completed")
    return W