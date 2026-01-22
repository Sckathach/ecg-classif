import os
import pickle
from typing import Literal

import networkx as nx
import numpy as np
import scipy
from jaxtyping import Float
from loguru import logger


def load_subject_data(
    band: Literal["ALPHA", "BETA", "THETA", "DELTA"] = "ALPHA",
    group: Literal["SCI", "MCI", "AD"] = "MCI",
    subject_id: int = 1,
    data_root: str = "data/",
) -> Float[np.ndarray, "30 30"]:
    filename = f"{subject_id}.mat"
    filepath = os.path.join(data_root, band, group, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        mat_data = scipy.io.loadmat(filepath)
        keys = [k for k in mat_data if not k.startswith("__")]
        matrix = mat_data[keys[0]]
        return matrix

    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {e}") from e


def extract_vector(matrix: Float[np.ndarray, "30 30"]) -> Float[np.ndarray, "435"]:
    # Upper triangle: 30*29/2 = 435 features

    upper_tri_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_tri_indices]


def extract_regional_means(
    matrix: Float[np.ndarray, "30 30"],
) -> Float[np.ndarray, "30"]:
    return np.mean(matrix, axis=1)


def extract_graph_metrics(
    matrix: Float[np.ndarray, "30 30"], threshold: float = 0.2
) -> Float[np.ndarray, "6"]:
    weights = matrix.flatten()
    weights.sort()

    # Index for the top X percent
    # e.g., if threshold is 0.2 (20%), we want the value at 80th percentile.
    cutoff_index = int(len(weights) * (1 - threshold))
    cutoff_index = max(0, min(cutoff_index, len(weights) - 1))
    cutoff_value = weights[cutoff_index]

    adj_matrix = (matrix >= cutoff_value).astype(int)
    G = nx.from_numpy_array(adj_matrix)

    # 1. Mean degree
    degrees = np.array([d for n, d in G.degree()])  # type: ignore
    mean_degree = np.mean(degrees)

    # 2. Clustering coefficient
    clustering_coeffs = nx.clustering(G)
    mean_clustering = np.mean(list(clustering_coeffs.values()))

    # Global efficiency
    global_eff = nx.global_efficiency(G)

    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    betweenness_vals = np.array(list(betweenness.values()))
    max_betweenness = np.max(betweenness_vals) if len(betweenness_vals) > 0 else 0
    mean_betweenness = np.mean(betweenness_vals) if len(betweenness_vals) > 0 else 0

    # Modularity (greedy Q-score)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity_q = nx.community.modularity(G, communities)
    except:  # noqa: E722
        modularity_q = 0.0

    return np.array(
        [
            mean_degree,
            mean_clustering,
            global_eff,
            max_betweenness,
            mean_betweenness,
            modularity_q,
        ]
    )


def generate_features(data_root: str = "data/", output_dir: str = "data/processed/"):
    os.makedirs(output_dir, exist_ok=True)

    dataset = {"vector": {}, "regional": {}, "graph": {}}
    group_map = {"SCI": 0, "MCI": 1, "AD": 2}

    for band in ["ALPHA", "BETA", "THETA", "DELTA"]:
        logger.info(f"Band: {band}...")

        X_vec, X_reg, X_graph = [], [], []
        y_list = []
        meta_list = []

        for group in ["SCI", "MCI", "AD"]:
            label = group_map[group]

            group_path = os.path.join(data_root, band, group)
            files = [f for f in os.listdir(group_path) if f.endswith(".mat")]
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            for f in files:
                subject_id = int(os.path.splitext(f)[0])
                try:
                    matrix = load_subject_data(
                        band,  # type: ignore
                        group,  # type: ignore
                        subject_id,
                        data_root=data_root,
                    )

                    vec = extract_vector(matrix)
                    X_vec.append(vec)

                    reg = extract_regional_means(matrix)
                    X_reg.append(reg)

                    graph = extract_graph_metrics(matrix, threshold=0.2)
                    X_graph.append(graph)

                    y_list.append(label)
                    meta_list.append((group, subject_id))

                except Exception as e:
                    logger.error(f"Error processing {band}-{group}-{subject_id}: {e}")

        # Convert to arrays and store in dataset
        dataset["vector"][band] = {
            "X": np.array(X_vec),
            "y": np.array(y_list),
            "meta": meta_list,
        }
        dataset["regional"][band] = {
            "X": np.array(X_reg),
            "y": np.array(y_list),
            "meta": meta_list,
        }
        dataset["graph"][band] = {
            "X": np.array(X_graph),
            "y": np.array(y_list),
            "meta": meta_list,
        }

        logger.info(f"  Shape vector: {dataset['vector'][band]['X'].shape}")
        logger.info(f"  Shape regional: {dataset['regional'][band]['X'].shape}")
        logger.info(f"  Shape graph: {dataset['graph'][band]['X'].shape}")

    output_file = os.path.join(output_dir, "features.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

    logger.info(f"Features saved to {output_file}")


if __name__ == "__main__":
    generate_features()
