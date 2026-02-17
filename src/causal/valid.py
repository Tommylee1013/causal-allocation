import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from econml.dml import LinearDML, CausalForestDML
from tqdm import tqdm

from .lingam import lingam_B_to_dag

def _proper_ancestors(G: nx.DiGraph, node: str) -> set[str]:
    return nx.ancestors(G, node)

def _proper_descendants(G: nx.DiGraph, node: str) -> set[str]:
    return nx.descendants(G, node)

def _as_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a

def find_mediators(G: nx.DiGraph, T: str, Y: str) -> set[str]:
    """
    Mediator: T -> ... -> M -> ... -> Y 형태로
    M이 (T의 descendant) 이면서 (Y의 ancestor) 인 노드
    """
    med = (_proper_descendants(G, T) & _proper_ancestors(G, Y)) - {T, Y}
    return med

def find_confounders(G: nx.DiGraph, T: str, Y: str) -> set[str]:
    """
    Confounder 후보: T와 Y의 공통 원인
    W ∈ Anc(T) ∩ Anc(Y)
    """
    conf = (_proper_ancestors(G, T) & _proper_ancestors(G, Y)) - {T, Y}
    return conf

def find_colliders_on_paths(G: nx.DiGraph, T: str, Y: str, max_simple_paths: int = 2000) -> set[str]:
    """
    Collider 후보:
    T와 Y 사이의 단순 경로(무방향으로 취급) 중에서,
    어떤 노드 v가 경로 상에서 양쪽 이웃 u,w로부터 u->v 그리고 w->v 가 동시에 성립하면 collider로 표시.
    (주의: 이는 '경로 상의 collider' 후보 탐지용 휴리스틱이며, 완전한 d-separation 판정은 아님)
    """
    UG = G.to_undirected()
    colliders = set()

    # 경로가 너무 많을 수 있으니 제한
    cnt = 0
    for path in nx.all_simple_paths(UG, source=T, target=Y):
        cnt += 1
        if cnt > max_simple_paths:
            break
        # 내부 노드에 대해 collider 여부 체크
        for i in range(1, len(path) - 1):
            u, v, w = path[i - 1], path[i], path[i + 1]
            if G.has_edge(u, v) and G.has_edge(w, v):
                colliders.add(v)
    return colliders

def classify_nodes_for_pair(G: nx.DiGraph, T: str, Y: str) -> dict[str, set[str]]:
    """
    Pair별로 confounder/mediator/collider 후보를 분리.
    우선순위:
    - mediator는 '조절하면 direct effect'가 되므로 별도 관리
    - collider는 절대 조절하지 않도록 별도 관리
    - confounder는 조절 후보
    """
    mediators = find_mediators(G, T, Y)
    confounders = find_confounders(G, T, Y) - mediators
    colliders = find_colliders_on_paths(G, T, Y) - mediators - confounders - {T, Y}

    return {
        "confounders": confounders,
        "mediators": mediators,
        "colliders": colliders,
    }

def run_dml_effect(
    df: pd.DataFrame,
    T: str,
    Y: str,
    confounders: list[str],
    mediators: list[str],
    effect_type: str = "total",
    discrete_treatment: bool = False,
    model_y=None,
    model_t=None,
    final_model: str = "forest",
    random_state: int = 0,
):
    # y, t
    y = df[Y].to_numpy()
    t = df[T].to_numpy()

    # X: confounders만 (요청대로 collider/mediator 제외)
    if confounders and len(confounders) > 0:
        X = df[confounders].to_numpy()
        X = _as_2d(X)
    else:
        # CausalForestDML은 X=None 불가 → 더미 상수열로 대체
        X = np.ones((len(df), 1), dtype=float)

    # W: effect_type="direct"일 때만 mediator를 W로 넣음(총효과면 None)
    if effect_type == "direct" and mediators and len(mediators) > 0:
        W = df[mediators].to_numpy()
        W = _as_2d(W)
    else:
        W = None

    # 기본 모델 세팅(외부에서 안 넣으면 내부에서 간단히 채움)
    if model_y is None or model_t is None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV

        if final_model == "forest":
            if model_y is None:
                model_y = RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=10, random_state=random_state, n_jobs=-1
                )
            if model_t is None:
                model_t = RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=10, random_state=random_state, n_jobs=-1
                )
        else:
            if model_y is None:
                model_y = LassoCV(cv=5, random_state=random_state)
            if model_t is None:
                model_t = LassoCV(cv=5, random_state=random_state)

    # Estimator
    if final_model == "forest":
        from econml.dml import CausalForestDML
        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            n_estimators=500,
            min_samples_leaf=10,
            random_state=random_state,
        )
    else:
        from econml.dml import LinearDML
        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            random_state=random_state,
        )

    est.fit(y, t, X=X, W=W)

    # ATE 요약
    eff = est.effect(X=X)
    ate = float(np.mean(eff))

    # 간단 CI(가능할 때만)
    ci95 = None
    try:
        inf = est.effect_interval(X=X, alpha=0.05)
        lo = float(np.mean(inf[0]))
        hi = float(np.mean(inf[1]))
        ci95 = (lo, hi)
    except Exception:
        pass

    return {"ate": ate, "ci95": ci95, "n_confounders": len(confounders), "n_mediators": len(mediators)}

def automated_dag_dml_pipeline(
    final_feature_matrix: pd.DataFrame,
    B_lingam: np.ndarray,
    macro_vars: list[str],
    cluster_vars: list[str],
    forbid_cluster_to_macro: bool = True,
    standardize: bool = True,
    effect_type: str = "total",     # "total" or "direct"
    final_model: str = "linear",    # "linear" or "forest"
    random_state: int = 0,
):
    """
    요구사항 반영:
    - 초기 DAG: LiNGAM B로 생성
    - 규칙: cluster1~3 -> macro 영향 금지 (원하면 간선 제거)
    - collider는 제어하지 않음: confounders만 X로 사용
    - mediator는 effect_type에 따라 (direct 효과일 때만) W로 넣음
    """

    df = final_feature_matrix.copy()

    # 표준화(권장: DML에서 모델 안정성)
    if standardize:
        scaler = StandardScaler()
        df.loc[:, :] = scaler.fit_transform(df.values)

    var_names = list(df.columns)
    G = lingam_B_to_dag(B_lingam, var_names)

    # 구조 제약: cluster -> macro 금지
    if forbid_cluster_to_macro:
        for c in cluster_vars:
            for m in macro_vars:
                if G.has_edge(c, m):
                    G.remove_edge(c, m)

    # 모든 (T, Y)쌍에 대해 분류 + DML
    results = []
    for T in tqdm(var_names):
        for Y in var_names:
            if T == Y:
                continue

            # 당신의 설정: cluster는 macro에 영향을 주지 않는다고 했으므로,
            # T가 cluster이고 Y가 macro면 스킵(또는 강제로 0으로 처리)
            if forbid_cluster_to_macro and (T in cluster_vars) and (Y in macro_vars):
                continue

            cls = classify_nodes_for_pair(G, T, Y)
            conf = sorted(list(cls["confounders"]))
            med = sorted(list(cls["mediators"]))
            col = sorted(list(cls["colliders"]))

            # collider는 절대 조절 변수에 넣지 않음
            # confounder만 X로 사용
            dml_out = run_dml_effect(
                df=df,
                T=T,
                Y=Y,
                confounders=conf,
                mediators=med,
                effect_type=effect_type,
                discrete_treatment=False,
                model_y=None,
                model_t=None,
                final_model=final_model,
                random_state=random_state,
            )

            results.append({
                "T": T,
                "Y": Y,
                "n_confounders": len(conf),
                "n_mediators": len(med),
                "n_colliders": len(col),
                "confounders": conf,
                "mediators": med,
                "colliders": col,
                "ate": dml_out["ate"],
                "ci95": dml_out["ci95"],
            })

    results_df = pd.DataFrame(results).sort_values(["T", "Y"]).reset_index(drop=True)
    return results_df, G

def list_edges(G, sources, targets):
    out = []
    for s in sources:
        for t in targets:
            if G.has_edge(s, t):
                out.append((s, t, G[s][t].get("weight", np.nan)))
    return sorted(out, key=lambda x: -abs(x[2]))

def enforce_min_macro_parents_from_B(G, B, cols, macro_vars, cluster_vars, k=1):
    """
    B[i, j] = i -> j weight (사용자 코드 기준)
    그래프에 Macro->Cluster 엣지가 0개인 cluster가 있으면,
    |B[m, c]|가 큰 순으로 k개를 추가
    """
    col_to_idx = {c:i for i,c in enumerate(cols)}
    H = G.copy()

    for c in cluster_vars:
        if c not in col_to_idx:
            continue
        macro_parents = [p for p in H.predecessors(c) if p in macro_vars]
        if len(macro_parents) >= k:
            continue

        ci = col_to_idx[c]
        cand = []
        for m in macro_vars:
            if m not in col_to_idx:
                continue
            mi = col_to_idx[m]
            w = float(B[mi, ci])
            if abs(w) > 0:
                cand.append((m, c, w))

        cand = sorted(cand, key=lambda x: -abs(x[2]))
        need = k - len(macro_parents)
        for m, c, w in cand[:need]:
            H.add_edge(m, c, weight=w)

    return H

def prune_in_edges_topk(G, targets, k=3):
    H = G.copy()
    for t in targets:
        in_edges = [(u, v, H[u][v].get("weight", 0.0)) for u, v in H.in_edges(t)]
        in_edges = sorted(in_edges, key=lambda x: -abs(x[2]))
        keep = set((u, v) for u, v, _ in in_edges[:k])
        for u, v, _ in in_edges[k:]:
            if (u, v) not in keep:
                H.remove_edge(u, v)
    return H