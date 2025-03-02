import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_similarity_matrix(file_path: str) -> np.ndarray:
    """Load and return the similarity matrix from a file."""
    logging.info(f"Loading similarity matrix from {file_path}")
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    matrix = np.round(matrix, decimals=2)
    logging.info(f"Matrix shape: {matrix.shape}")
    logging.info(f"Non-zero elements: {np.count_nonzero(matrix)}")
    logging.info(f"Matrix value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
    return matrix


def create_heatmap(df: pd.DataFrame,
                   output_file: str,
                   title: str,
                   xlabel="Similarity Percentile",
                   ylabel="Centrality Percentile"):
    """Create and save a heatmap visualization."""
    logging.info(f"Creating heatmap: {title}")
    logging.info(f"Output file: {output_file}")

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
    logging.info("Heatmap created successfully")


def calculate_thresholds(percentile_range: tuple = (50, 100),
                         step: int = 5):
    """
    Calculate thresholds for cross-platform network analysis.
    """
    logging.info("\n=== Starting Cross-Platform Threshold Calculation ===")
    logging.info(f"Percentile range: {percentile_range}")
    logging.info(f"Step size: {step}")

    # Set up paths
    base_dir = "../networks/cross_platform"
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Working directory: {base_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Load similarity matrix
    similarity_matrix = load_similarity_matrix(os.path.join(base_dir, "author_similarity_matrix.npz"))

    # Load author mappings to get platform information
    logging.info("Loading author mappings...")
    with open(os.path.join(base_dir, "author_mappings.json"), 'r') as f:
        author_data = json.load(f)
    logging.info(f"Loaded author mappings with {len(author_data['author_to_idx'])} authors")

    logging.info("Creating initial graph...")
    num_nodes = similarity_matrix.shape[0]
    G_full = nx.Graph()

    # Create full graph
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > 0:
                G_full.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_count += 1

    logging.info(f"Initial graph created with {num_nodes} nodes and {edge_count} edges")
    logging.info(f"Graph density: {nx.density(G_full):.4f}")

    logging.info("Calculating centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
        centrality_values = np.array(list(eigenvector_centrality.values()))
        logging.info("Using eigenvector centrality")
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality failed to converge, using degree centrality instead")
        eigenvector_centrality = nx.degree_centrality(G_full)
        centrality_values = np.array(list(eigenvector_centrality.values()))

    logging.info(f"Centrality value range: [{centrality_values.min():.4f}, {centrality_values.max():.4f}]")

    percentiles = np.arange(percentile_range[0], percentile_range[1], step)
    density_matrix = {}
    graph_data = {}

    logging.info("Calculating density matrices...")
    logging.info(f"Processing {len(percentiles)} centrality and similarity percentiles")

    # Calculate density matrices for different threshold combinations
    total_combinations = len(percentiles) * len(percentiles)
    with tqdm(total=total_combinations, desc="Processing thresholds") as pbar:
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

    logging.info("Creating density DataFrame...")
    df_density = pd.DataFrame.from_dict(density_matrix, orient="index").sort_index(ascending=False)
    df_density.columns = sorted(df_density.columns)
    logging.info(f"Density matrix shape: {df_density.shape}")

    logging.info("Calculating increments between centrality levels...")
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
                logging.info(f"New maximum increment found:")
                logging.info(f"Similarity percentile: {sim_p}")
                logging.info(f"Centrality percentile: {cent_current}")
                logging.info(f"Increment value: {increment:.4f}")

            if cent_current not in increments_matrix:
                increments_matrix[cent_current] = {}
            increments_matrix[cent_current][sim_p] = float(increment)

    df_increments = pd.DataFrame.from_dict(increments_matrix, orient="index").sort_index(ascending=False)
    df_increments.columns = sorted(df_increments.columns)
    logging.info(f"Increments matrix shape: {df_increments.shape}")

    logging.info("Creating heatmaps...")
    create_heatmap(df_density,
                   os.path.join(output_dir, "density_heatmap.png"),
                   "Cross-Platform Network Density by Threshold Percentiles")

    create_heatmap(df_increments,
                   os.path.join(output_dir, "increments_heatmap.png"),
                   "Cross-Platform Density Increments Between Centrality Levels")

    logging.info("Saving thresholds...")
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
    logging.info(f"Thresholds saved to: {thresholds_file}")

    # Print final summary statistics
    logging.info("\n=== Final Results ===")
    logging.info(f"Optimal similarity percentile: {global_max_coords[0]}")
    logging.info(f"Optimal centrality percentile: {global_max_coords[1]}")
    logging.info(f"Maximum density increment: {global_max_increment:.4f}")
    logging.info("Analysis completed successfully")

    return global_max_coords


if __name__ == "__main__":
    calculate_thresholds()