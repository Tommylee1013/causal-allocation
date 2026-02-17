import numpy as np
import pandas as pd
import statsmodels.api as sm
import networkx as nx

from tigramite.data_processing import DataFrame as TigDataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tqdm import tqdm

def make_link_assumptions(
        var_names,
        tau_max,
        macro_vars,
        cluster_vars
    ) :
    """
    tigramite PCMCI의 link_assumptions:
    link_assumptions[j][i][tau] = "?"(allow) or "0"(forbid)

    X^i_{t-tau} -> X^j_t
    """
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    macro_idx   = {name_to_idx[n] for n in macro_vars if n in name_to_idx}
    cluster_idx = {name_to_idx[n] for n in cluster_vars if n in name_to_idx}

    n = len(var_names)

    # 1) 기본: 모든 "시차 링크" 허용 (튜플 key: (source_idx, -tau))
    link_assumptions = {}
    for j in range(n):
        links_j = {}
        for i in range(n):
            for tau in range(1, tau_max + 1):
                links_j[(i, -tau)] = "o->"  # lagged link (방향은 시간상 i_{t-tau} -> j_t)
        link_assumptions[j] = links_j

    # 2) 제약: Cluster -> Macro 금지 (모든 lag에서 삭제)
    for j in macro_idx:
        for i in cluster_idx:
            for tau in range(1, tau_max + 1):
                link_assumptions[j].pop((i, -tau), None)

    return link_assumptions

def run_pcmci(final_feature_matrix: pd.DataFrame,
              tau_max: int = 5,
              pc_alpha: float = 0.1,
              alpha_level: float = 0.05,
              macro_vars=None,
              cluster_vars=None):

    df = final_feature_matrix.copy()

    # tigramite는 numpy array를 받음 (결측 제거 권장)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    var_names = list(df.columns)

    if macro_vars is None:
        macro_vars = ["USOIL", "TBILL", "COPPER", "DXY", "HYS", "BEI", "VIX"]
    if cluster_vars is None:
        cluster_vars = [c for c in var_names if c.lower().startswith("cluster")]

    tig_df = TigDataFrame(df.values, var_names=var_names)

    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=parcorr, verbosity=1)

    link_assumptions = make_link_assumptions(
        var_names=var_names,
        tau_max=tau_max,
        macro_vars=macro_vars,
        cluster_vars=cluster_vars
    )

    results = pcmci.run_pcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        link_assumptions=link_assumptions
    )

    return results, var_names, df

def pcmci_to_nx(
        results,
        var_names,
        alpha_level=0.05
    ) -> nx.Graph:
    p = results["p_matrix"]      # shape: [N, N, tau_max+1] or [N, N, tau_max]
    val = results["val_matrix"]
    tau_max = p.shape[2] - 1

    G = nx.DiGraph()

    # 노드 생성 (t, t-1,...t-tau_max)
    for name in var_names:
        for lag in range(tau_max + 1):
            G.add_node(f"{name}@t-{lag}")

    for j, tgt in enumerate(var_names):
        for i, src in enumerate(var_names):
            for tau in range(1, tau_max + 1):  # tau=0(동시) 제외(원하면 포함 가능)
                if p[j, i, tau] <= alpha_level:
                    u = f"{src}@t-{tau}"
                    v = f"{tgt}@t-0"
                    G.add_edge(u, v, weight=float(val[j, i, tau]), pval=float(p[j, i, tau]))

    return G

def confounders_only(G: nx.DiGraph, treatment: str, outcome: str):
    """
    treatment: "X@t-k"
    outcome:   "Y@t-0"
    confounder = Ancestors(treatment) ∩ Ancestors(outcome)
    collider는 '공통결과'를 조건화할 때 생기므로, 공통원인만 고르면 collider conditioning을 크게 줄일 수 있음.
    """
    anc_t = nx.ancestors(G, treatment)
    anc_y = nx.ancestors(G, outcome)

    common = anc_t.intersection(anc_y)

    common = {z for z in common if z not in {treatment, outcome}}

    return sorted(common)

def estimate_effect_hac(df_raw: pd.DataFrame,
                        treatment_var: str,
                        outcome_var: str,
                        lags_for_treatment: int,
                        control_nodes: list,
                        hac_lags: int = 5):
    """
    df_raw: 원래 t 시계열 데이터프레임(열: USOIL..Cluster_3)
    treatment_var: 예) "VIX"
    outcome_var: 예) "Cluster_1"
    lags_for_treatment: 예) 1 => VIX(t-1)로 개입
    control_nodes: ["DXY@t-1", "TBILL@t-2", ...] 형태
    """

    df = df_raw.copy()

    # outcome: Y_t
    y = df[outcome_var]

    # treatment: X_{t-k}
    x = df[treatment_var].shift(lags_for_treatment)

    X = pd.DataFrame({"treat": x})

    # controls: Z_{t-l}
    for node in control_nodes:
        name, lag = node.split("@t-")
        lag = int(lag)
        X[node] = df[name].shift(lag)

    data = pd.concat([y, X], axis=1).dropna()
    y2 = data[outcome_var]
    X2 = sm.add_constant(data.drop(columns=[outcome_var]))

    model = sm.OLS(y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model

def automated_pmcmi_pipeline(
        final_feature_matrix: pd.DataFrame,
        macro_vars: list,
        cluster_vars: list,
        tau_max: int = 5,
        pc_alpha: float = 0.1,
        alpha_level: float = 0.05
    ):

    results, var_names, df_clean = run_pcmci(
        final_feature_matrix,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level
    )

    G = pcmci_to_nx(results, var_names, alpha_level=alpha_level)

    rows = []

    # treatment는 macro의 lagged value만 개입(예: t-1..t-tau_max)
    for X in tqdm(macro_vars):
        for Y in cluster_vars:
            for k in range(1, tau_max + 1):
                treat_node = f"{X}@t-{k}"
                out_node = f"{Y}@t-0"

                Z = confounders_only(G, treat_node, out_node)

                # HAC 회귀로 효과추정
                try:
                    m = estimate_effect_hac(
                        df_raw=df_clean,
                        treatment_var=X,
                        outcome_var=Y,
                        lags_for_treatment=k,
                        control_nodes=Z,
                        hac_lags=tau_max
                    )
                    rows.append({
                        "treatment": X,
                        "treat_lag": k,
                        "outcome": Y,
                        "n_controls": len(Z),
                        "controls": Z,
                        "coef": float(m.params.get("treat", np.nan)),
                        "tstat": float(m.tvalues.get("treat", np.nan)),
                        "pval": float(m.pvalues.get("treat", np.nan)),
                        "r2": float(m.rsquared),
                        "nobs": int(m.nobs),
                    })
                except Exception as e:
                    rows.append({
                        "treatment": X,
                        "treat_lag": k,
                        "outcome": Y,
                        "n_controls": len(Z),
                        "controls": Z,
                        "coef": np.nan,
                        "tstat": np.nan,
                        "pval": np.nan,
                        "r2": np.nan,
                        "nobs": 0,
                        "error": str(e),
                    })

    effects = pd.DataFrame(rows).sort_values(["outcome", "treatment", "treat_lag"])
    return results, G, effects