import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_similarity_matrix(file_path: str) -> np.ndarray:
    """Load and process similarity matrix from file."""
    logging.info(f"Loading similarity matrix from {file_path}")
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    matrix = np.round(matrix, decimals=2)
    logging.info(f"Loaded matrix with shape {matrix.shape}")
    return matrix


def create_heatmap(df: pd.DataFrame,
                   output_file: str,
                   title: str,
                   xlabel="Similarity Percentile",
                   ylabel="Centrality Percentile"):
    """Create and save heatmap visualization."""
    logging.info(f"Creating heatmap: {title}")
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
    logging.info(f"Saved heatmap to {output_file}")


def calculate_thresholds(platform: str,
                         percentile_range: tuple = (70, 100),
                         step: int = 2):
    """Calculate network thresholds and generate visualizations."""
    logging.info(f"Starting threshold calculation for platform: {platform}")
    logging.info(f"Percentile range: {percentile_range}, Step size: {step}")

    # Setup paths and load data
    matrix_path = f"../networks/{platform}/author_similarity_matrix.npz"
    if not os.path.exists(matrix_path):
        logging.error(f"Similarity matrix not found: {matrix_path}")
        raise FileNotFoundError(f"Similarity matrix not found: {matrix_path}")

    similarity_matrix = load_similarity_matrix(matrix_path)

    output_dir = f"../networks/{platform}/output"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Create initial graph
    logging.info("Creating initial graph")
    num_nodes = similarity_matrix.shape[0]
    G_full = nx.Graph()

    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > 0:
                G_full.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_count += 1

    logging.info(f"Created graph with {num_nodes} nodes and {edge_count} edges")

    # Calculate centrality
    logging.info("Calculating node centrality")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
        centrality_values = np.array(list(eigenvector_centrality.values()))
        logging.info("Using eigenvector centrality")
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality failed to converge, using degree centrality instead")
        eigenvector_centrality = nx.degree_centrality(G_full)
        centrality_values = np.array(list(eigenvector_centrality.values()))

    # Process percentiles
    percentiles = np.arange(percentile_range[0], percentile_range[1], step)
    logging.info(f"Processing {len(percentiles)} percentile combinations")

    density_matrix = {}
    graph_data = {}

    # Calculate densities for each threshold combination
    for cent_p in tqdm(percentiles, desc="Processing centrality percentiles"):
        cent_p_key = int(cent_p)
        density_matrix[cent_p_key] = {}
        graph_data[cent_p_key] = {}
        centrality_threshold = np.percentile(centrality_values, cent_p)

        for sim_p in percentiles:
            sim_p_key = int(sim_p)
            similarity_threshold = np.round(np.percentile(similarity_matrix[similarity_matrix > 0], sim_p), 2)

            # Store thresholds
            graph_data[cent_p_key][sim_p_key] = {
                'centrality_threshold': float(centrality_threshold),
                'similarity_threshold': float(similarity_threshold)
            }

            # Create filtered graph
            G_filtered = nx.Graph()
            for u, v, w in G_full.edges(data=True):
                if np.round(w['weight'], decimals=2) >= similarity_threshold:
                    if (eigenvector_centrality[u] >= centrality_threshold and
                            eigenvector_centrality[v] >= centrality_threshold):
                        G_filtered.add_edge(u, v, weight=w['weight'])

            # Calculate density
            if G_filtered.number_of_nodes() > 0:
                components = list(nx.connected_components(G_filtered))
                min_density = min(nx.density(G_filtered.subgraph(c))
                                  if len(G_filtered.subgraph(c)) > 1 else 0
                                  for c in components)
            else:
                min_density = 0

            density_matrix[cent_p_key][sim_p_key] = float(min_density)

    # Create density dataframe
    df_density = pd.DataFrame.from_dict(density_matrix, orient="index").sort_index(ascending=False)
    df_density.columns = sorted(df_density.columns)

    # Calculate increments
    logging.info("Calculating density increments")
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

            if cent_current not in increments_matrix:
                increments_matrix[cent_current] = {}
            increments_matrix[cent_current][sim_p] = float(increment)

    df_increments = pd.DataFrame.from_dict(increments_matrix, orient="index").sort_index(ascending=False)
    df_increments.columns = sorted(df_increments.columns)

    # Create visualizations
    logging.info("Generating visualizations")
    create_heatmap(df_density,
                   os.path.join(output_dir, "density_heatmap.png"),
                   f"{platform.capitalize()} Network Density by Threshold Percentiles")

    create_heatmap(df_increments,
                   os.path.join(output_dir, "increments_heatmap.png"),
                   f"{platform.capitalize()} Density Increments Between Centrality Levels")

    # Save results
    logging.info("Saving threshold data")
    thresholds_data = {
        'graph_data': graph_data,
        'optimal_coordinates': {
            'similarity_percentile': int(global_max_coords[0]),
            'centrality_percentile': int(global_max_coords[1])
        }
    }

    output_path = os.path.join(output_dir, "thresholds.json")
    with open(output_path, 'w') as f:
        json.dump(thresholds_data, f, indent=4)

    logging.info(f"Optimal thresholds found at similarity={global_max_coords[0]}, centrality={global_max_coords[1]}")
    logging.info(f"Results saved to {output_path}")
    logging.info("Threshold calculation completed successfully")
