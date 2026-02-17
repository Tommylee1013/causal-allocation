import numpy as np
import pandas as pd
import networkx as nx

from lingam import DirectLiNGAM
from sklearn.preprocessing import StandardScaler
from src.utils.func import base_name

def collapse_to_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    @t-0/@t-1 같은 시차 표기를 가진 컬럼을 정적(시차 제거) 컬럼으로 변환.
    - @t-0가 있으면 @t-0만 사용
    - @t-0가 없으면 그냥 원 컬럼 사용
    - 동일 base_name 중복 발생 시 첫 번째만 사용(필요하면 사용자 정책으로 바꾸세요)
    """
    cols = list(df.columns)
    has_lag = any("@t-" in c for c in cols)

    if not has_lag:
        return df.copy()

    # 우선 @t-0만 선택
    t0_cols = [c for c in cols if c.endswith("@t-0")]
    if len(t0_cols) == 0:
        # @t-0가 없다면 그냥 base_name별로 첫 컬럼 사용
        chosen = {}
        for c in cols:
            b = base_name(c)
            if b not in chosen:
                chosen[b] = c
        out = df[list(chosen.values())].copy()
        out.columns = list(chosen.keys())
        return out

    out = df[t0_cols].copy()
    out.columns = [base_name(c) for c in t0_cols]

    # base 중복 제거
    out = out.loc[:, ~out.columns.duplicated()]
    return out

def make_prior_knowledge(
    columns: list[str],
    macro_vars: list[str],
    cluster_vars: list[str],
    forbid_cluster_to_macro: bool = True,
    extra_forbidden_edges: list[tuple[str, str]] | None = None,
):
    """
    lingam 버전 차이를 흡수:
      - PriorKnowledge 객체 반환 버전
      - numpy.ndarray 반환 버전 모두 지원
    """
    col_to_idx = {c: i for i, c in enumerate(columns)}
    p = len(columns)

    # 1) 객체 방식 시도
    try:
        from lingam.utils import PriorKnowledge
        pk = PriorKnowledge(n_variables=p)

        def forbid(i, j):
            pk.add_forbidden_edge(i, j)

        is_object = True

    except Exception:
        # 2) 행렬 방식 (numpy.ndarray)
        from lingam.utils import make_prior_knowledge as _make_pk
        pk = _make_pk(n_variables=p)

        def forbid(i, j):
            pk[i, j] = -1

        is_object = False

    def forbid_by_name(src: str, dst: str):
        if src in col_to_idx and dst in col_to_idx:
            forbid(col_to_idx[src], col_to_idx[dst])

    # Cluster -> Macro 금지
    if forbid_cluster_to_macro:
        for c in cluster_vars:
            for m in macro_vars:
                forbid_by_name(c, m)

    # 추가 금지 간선
    if extra_forbidden_edges:
        for src, dst in extra_forbidden_edges:
            forbid_by_name(src, dst)

    return pk

def build_forbid_edges(
    cols: list[str],
    macro_vars: list[str],
    tbill_policy: str = "relaxed",  # "strict" or "relaxed"
):
    """
    tbill_policy
      - "strict": TBILL을 항상 원인으로 유도 (others -> TBILL 금지)
      - "relaxed": TBILL이 일부 변수의 결과일 가능성 허용 (macro들만 -> TBILL 금지)

    LiNGAM prior는 허용을 강제할 수 없으므로,
    방향성(예: TBILL이 원인)은 반대방향 금지로 구현합니다.
    """
    forbid_edges: list[tuple[str, str]] = []

    # (3) 명시 금지
    # (USOIL, COPPER) -> DXY 금지
    if "DXY" in cols:
        for src in ("USOIL", "COPPER"):
            if src in cols:
                forbid_edges.append((src, "DXY"))

    # HYS -> VIX 금지
    if "HYS" in cols and "VIX" in cols:
        forbid_edges.append(("HYS", "VIX"))

    # (1) TBILL 관련: 강제(Strict) vs 완화(Relaxed)
    if "TBILL" in cols:
        if tbill_policy == "strict":
            # others -> TBILL 금지 (가장 강함)
            for v in cols:
                if v != "TBILL":
                    forbid_edges.append((v, "TBILL"))
        elif tbill_policy == "relaxed":
            # macro들만 -> TBILL 금지 (완화)
            for v in macro_vars:
                if v in cols and v != "TBILL":
                    forbid_edges.append((v, "TBILL"))
        else:
            raise ValueError("tbill_policy must be 'strict' or 'relaxed'.")

    return forbid_edges

def fit_lingam_static(
        feature_matrix: pd.DataFrame,
        macro_vars : list | tuple,
        cluster_prefix = "Cluster_",
        standardize: bool = True,
        random_state: int = 7,
        tbill_policy: str = "relaxed",   # "strict" or "relaxed"
    ) -> tuple :
    """
    시차 제거 + prior 적용 + DirectLiNGAM 적합.

    포함된 설계 원칙
      1) TBILL 방향성 제약은 '허용 강제'가 아니라 '반대방향 금지'로 구현
         - strict: others -> TBILL 금지 (TBILL root 강제에 가까움)
         - relaxed: macro들만 -> TBILL 금지 (TBILL이 cluster/기타의 결과일 가능성 일부 허용)
      2) LiNGAM prior는 허용을 강제하기 어렵고, 금지로만 탐색공간을 줄입니다.

    반환:
      model, B, order, df_used
      - B[i, j] = i -> j 가중치 (lingam adjacency_matrix_ convention)
    """
    df0 = collapse_to_static(feature_matrix)

    macro_vars = list(macro_vars)
    cluster_vars = [c for c in df0.columns if str(c).startswith(cluster_prefix)]

    df = df0.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()

    X = df.values
    if standardize:
        X = StandardScaler().fit_transform(X)

    cols = list(df.columns)
    forbid_edges = build_forbid_edges(
        cols=cols,
        macro_vars=macro_vars,
        tbill_policy=tbill_policy,
    )

    pk = make_prior_knowledge(
        columns=cols,
        macro_vars=macro_vars,
        cluster_vars=cluster_vars,
        forbid_cluster_to_macro=True,
        extra_forbidden_edges=forbid_edges,
    )

    model = DirectLiNGAM(prior_knowledge=pk, random_state=random_state)
    model.fit(X)

    B = model.adjacency_matrix_.copy()   # B[i, j] = i -> j
    order = model.causal_order_

    return model, B, order, df

def adjacency_to_digraph(
        B: np.ndarray,
        columns: list[str],
        thresh: float = 1e-6
    ) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(columns)

    p = len(columns)
    for i in range(p):
        for j in range(p):
            w = B[i, j]
            if abs(w) > thresh:
                G.add_edge(columns[i], columns[j], weight=float(w))
    return G

def lingam_B_to_dag(B: np.ndarray, var_names: list[str], weight_threshold: float = 1e-8) -> nx.DiGraph:
    """
    B[i, j] : (j -> i) 계수라고 가정하는 표준 SEM 표기( x = Bx + e )
    즉, B[target, source] = weight
    """
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    p = len(var_names)
    for target in range(p):
        for source in range(p):
            w = B[target, source]
            if abs(w) > weight_threshold:
                G.add_edge(var_names[source], var_names[target], weight=float(w))
    return G