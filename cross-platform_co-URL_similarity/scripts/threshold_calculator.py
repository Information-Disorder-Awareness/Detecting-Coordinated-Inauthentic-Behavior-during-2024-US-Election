import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
from tqdm import tqdm


def load_similarity_matrix(file_path: str) -> np.ndarray:
    print(f"\nLoading similarity matrix from {file_path}")
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    matrix = np.round(matrix, decimals=2)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Matrix value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
    return matrix


def create_heatmap(df: pd.DataFrame,
                   output_file: str,
                   title: str,
                   xlabel="Similarity Percentile",
                   ylabel="Centrality Percentile"):
    print(f"\nCreating heatmap: {title}")
    print(f"Output file: {output_file}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Value'},
                square=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("Heatmap created successfully")


def calculate_cross_platform_thresholds(percentile_range: tuple = (70, 100),
                                        step: int = 2):
    """
    Calculate thresholds for cross-platform network analysis.
    """
    print("\n=== Starting Cross-Platform Threshold Calculation ===")
    print(f"Percentile range: {percentile_range}")
    print(f"Step size: {step}")

    # Set up paths
    base_dir = "../networks/cross_platform"
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nWorking directory: {base_dir}")
    print(f"Output directory: {output_dir}")

    # Load similarity matrix
    similarity_matrix = load_similarity_matrix(os.path.join(base_dir, "author_similarity_matrix.npz"))

    # Load author mappings to get platform information
    print("\nLoading author mappings...")
    with open(os.path.join(base_dir, "author_mappings.json"), 'r') as f:
        author_data = json.load(f)
    print(f"Loaded author mappings with {len(author_data['author_to_idx'])} authors")

    print("\nCreating initial graph...")
    num_nodes = similarity_matrix.shape[0]
    G_full = nx.Graph()

    # Create full graph
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > 0:
                G_full.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_count += 1

    print(f"Initial graph created with {num_nodes} nodes and {edge_count} edges")
    print(f"Graph density: {nx.density(G_full):.4f}")

    print("\nCalculating centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
        centrality_values = np.array(list(eigenvector_centrality.values()))
        print("Using eigenvector centrality")
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality failed to converge, using degree centrality instead")
        eigenvector_centrality = nx.degree_centrality(G_full)
        centrality_values = np.array(list(eigenvector_centrality.values()))

    print(f"Centrality value range: [{centrality_values.min():.4f}, {centrality_values.max():.4f}]")

    percentiles = np.arange(percentile_range[0], percentile_range[1], step)
    density_matrix = {}
    graph_data = {}

    print("\nCalculating density matrices...")
    print(f"Processing {len(percentiles)} centrality and similarity percentiles")

    # Calculate density matrices for different threshold combinations
    with tqdm(total=len(percentiles) * len(percentiles)) as pbar:
        for cent_p in percentiles:
            cent_p_key = int(cent_p)
            density_matrix[cent_p_key] = {}
            graph_data[cent_p_key] = {}
            centrality_threshold = np.percentile(centrality_values, cent_p)

            for sim_p in percentiles:
                sim_p_key = int(sim_p)
                similarity_threshold = np.round(np.percentile(similarity_matrix[similarity_matrix > 0], sim_p), 2)

                graph_data[cent_p_key][sim_p_key] = {
                    'centrality_threshold': float(centrality_threshold),
                    'similarity_threshold': float(similarity_threshold)
                }

                G_filtered = nx.Graph()

                for u, v, w in G_full.edges(data=True):
                    if np.round(w['weight'], decimals=2) >= similarity_threshold:
                        if (eigenvector_centrality[u] >= centrality_threshold and
                                eigenvector_centrality[v] >= centrality_threshold):
                            G_filtered.add_edge(u, v, weight=w['weight'])

                if G_filtered.number_of_nodes() > 0:
                    components = list(nx.connected_components(G_filtered))
                    min_density = min(nx.density(G_filtered.subgraph(c))
                                      if len(G_filtered.subgraph(c)) > 1 else 0
                                      for c in components)
                else:
                    min_density = 0

                density_matrix[cent_p_key][sim_p_key] = float(min_density)
                pbar.update(1)

    print("\nCreating density DataFrame...")
    df_density = pd.DataFrame.from_dict(density_matrix, orient="index").sort_index(ascending=False)
    df_density.columns = sorted(df_density.columns)
    print("Density matrix shape:", df_density.shape)

    print("\nCalculating increments between centrality levels...")
    increments_matrix = {}
    global_max_increment = -float('inf')
    global_max_coords = None

    for sim_p in percentiles:
        sim_p = int(sim_p)
        centrality_values = sorted(df_density.index, reverse=True)
        for i in range(len(centrality_values) - 1):
            cent_current = centrality_values[i]
            cent_next = centrality_values[i + 1]

            increment = df_density.loc[cent_current, sim_p] - df_density.loc[cent_next, sim_p]

            if increment > global_max_increment:
                global_max_increment = increment
                global_max_coords = (sim_p, cent_current)
                print(f"\nNew maximum increment found:")
                print(f"Similarity percentile: {sim_p}")
                print(f"Centrality percentile: {cent_current}")
                print(f"Increment value: {increment:.4f}")

            if cent_current not in increments_matrix:
                increments_matrix[cent_current] = {}
            increments_matrix[cent_current][sim_p] = float(increment)

    df_increments = pd.DataFrame.from_dict(increments_matrix, orient="index").sort_index(ascending=False)
    df_increments.columns = sorted(df_increments.columns)
    print("\nIncrements matrix shape:", df_increments.shape)

    print("\nCreating heatmaps...")
    create_heatmap(df_density,
                   os.path.join(output_dir, "density_heatmap.png"),
                   "Cross-Platform Network Density by Threshold Percentiles")

    create_heatmap(df_increments,
                   os.path.join(output_dir, "increments_heatmap.png"),
                   "Cross-Platform Density Increments Between Centrality Levels")

    print("\nSaving thresholds...")
    thresholds_data = {
        'graph_data': graph_data,
        'optimal_coordinates': {
            'similarity_percentile': int(global_max_coords[0]),
            'centrality_percentile': int(global_max_coords[1])
        }
    }

    thresholds_file = os.path.join(output_dir, "thresholds.json")
    with open(thresholds_file, 'w') as f:
        json.dump(thresholds_data, f, indent=4)
    print(f"Thresholds saved to: {thresholds_file}")

    # Print final summary statistics
    print("\n=== Final Results ===")
    print(f"Optimal similarity percentile: {global_max_coords[0]}")
    print(f"Optimal centrality percentile: {global_max_coords[1]}")
    print(f"Maximum density increment: {global_max_increment:.4f}")
    print("\nAnalysis completed successfully")

    return global_max_coords


if __name__ == "__main__":
    calculate_cross_platform_thresholds()