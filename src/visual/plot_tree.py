import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout

def plot_graph_tree(
        G: nx.DiGraph,
        macro_vars=None,
        cluster_prefix="Cluster_",
        figsize=(14, 8)
    ) -> None :
    if macro_vars is None:
        macro_vars = ["USOIL", "TBILL", "COPPER", "DXY", "HYS", "BEI", "VIX"]

    nodes = list(G.nodes())

    macro_nodes = [n for n in nodes if any(n.startswith(m) for m in macro_vars)]
    cluster_nodes = [n for n in nodes if n.startswith(cluster_prefix)]
    other_nodes = [n for n in nodes if n not in macro_nodes + cluster_nodes]

    pos = {}

    # 좌측: 매크로
    y_macro = np.linspace(1, -1, len(macro_nodes))
    for i, n in enumerate(sorted(macro_nodes)):
        pos[n] = (0.0, y_macro[i])

    # 중앙: 기타 (있으면)
    y_mid = np.linspace(1, -1, len(other_nodes)) if other_nodes else []
    for i, n in enumerate(sorted(other_nodes)):
        pos[n] = (1.5, y_mid[i])

    # 우측: 클러스터
    y_cluster = np.linspace(1, -1, len(cluster_nodes))
    for i, n in enumerate(sorted(cluster_nodes)):
        pos[n] = (3.0, y_cluster[i])

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, nodelist=macro_nodes, node_size=1200)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_size=1200)

    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=1.2, alpha=0.7)

    plt.axis("off")
    plt.show()

def plot_treeish(
        G : nx.DiGraph,
        figsize : tuple = (14, 9),
        font_size : int = 9,
        min_width : float = 0.5,
        max_width : float = 4.0,
        show_weights : bool = True,
        arrowsize : int = 20,
        arrowstyle = "-|>"
    ) -> None :

    pos = graphviz_layout(G, prog="dot")  # 계층 DAG 레이아웃

    edges = list(G.edges(data=True))
    if len(edges) > 0:
        w_abs = np.array([abs(d.get("weight", 1.0)) for _, _, d in edges])
        w_min, w_max = w_abs.min(), w_abs.max()
        if w_max > w_min:
            widths = min_width + (w_abs - w_min) / (w_max - w_min) * (max_width - min_width)
        else:
            widths = np.full_like(w_abs, (min_width + max_width) / 2)
    else:
        widths = []

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle=arrowstyle,
        arrowsize=arrowsize,   # 여기
        width=widths
    )

    if show_weights and len(edges) > 0:
        edge_labels = {(u, v): f"{d.get('weight', 0.0):.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.show()

def plot_dag_dot(
        G: nx.DiGraph,
        root: str | None = None,
        rankdir: str = "LR",          # "LR" 좌→우, "TB" 위→아래
        figsize=(14, 8),
        node_size=900,
        font_size=9,
        min_width=0.5,
        max_width=4.0,
        show_weights=True,
        weight_fmt="{:.2f}",
        prune_abs_weight: float | None = None,   # 예: 0.05 주면 약한 엣지 제거
    ) -> None :
    # (선택) 약한 엣지 제거
    H = G.copy()
    if prune_abs_weight is not None:
        drop = []
        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            if abs(w) < prune_abs_weight:
                drop.append((u, v))
        H.remove_edges_from(drop)

    # root 기준으로 "계층"을 더 강하게 만들고 싶으면:
    # root를 주면 root에서 도달 가능한 서브그래프를 우선 사용
    if root is not None and root in H:
        reachable = set(nx.descendants(H, root)) | {root}
        # root와 무관한 노드도 같이 보려면 이 줄을 주석처리
        # H = H.subgraph(reachable).copy()

    # Graphviz dot 레이아웃
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(
            H,
            prog="dot",
            args=f"-Grankdir={rankdir} -Goverlap=false -Gsplines=true -Gsep=0.6"
        )
    except Exception as e:
        raise ImportError(
            "Graphviz 레이아웃을 쓰려면 pygraphviz(또는 pydot)가 필요합니다. "
            "conda 환경이면 `conda install -c conda-forge pygraphviz`를 권장합니다."
        ) from e

    # 엣지 두께 스케일
    edges = list(H.edges(data=True))
    if edges:
        w_abs = np.array([abs(d.get("weight", 1.0)) for _, _, d in edges], dtype=float)
        w_min, w_max = float(w_abs.min()), float(w_abs.max())
        if w_max > w_min:
            widths = min_width + (w_abs - w_min) / (w_max - w_min) * (max_width - min_width)
        else:
            widths = np.full_like(w_abs, (min_width + max_width) / 2.0)
    else:
        widths = []

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(H, pos, node_size=node_size)
    nx.draw_networkx_labels(H, pos, font_size=font_size)

    nx.draw_networkx_edges(
        H, pos,
        arrows=True, arrowstyle="->",
        width=widths,
        connectionstyle="arc3,rad=0.08"  # 약간 휘게 해서 겹침 완화
    )

    if show_weights and edges:
        edge_labels = {(u, v): weight_fmt.format(d.get("weight", 0.0)) for u, v, d in edges}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.show()