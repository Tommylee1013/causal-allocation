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