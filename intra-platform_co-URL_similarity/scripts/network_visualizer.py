import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_label(text):
    """Clean label text by removing emojis and problematic Unicode characters"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    cleaned_text = emoji_pattern.sub('', text)
    cleaned_text = ''.join(c for c in cleaned_text if c.isascii() or c.isspace())
    return cleaned_text.strip()


def load_similarity_matrix(file_path: str) -> np.ndarray:
    """Load and process similarity matrix from file."""
    logging.info(f"Loading similarity matrix from {file_path}")
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    matrix = np.round(matrix, decimals=2)
    logging.info(f"Loaded matrix with shape {matrix.shape}")
    return matrix


def load_author_mappings(platform: str) -> dict:
    """Load author mappings from the JSON file"""
    mapping_path = f"../networks/{platform}/author_url_mappings.json"
    logging.info(f"Loading author mappings from {mapping_path}")
    try:
        with open(mapping_path, 'r') as f:
            data = json.load(f)
            mappings = data['author_to_idx']
            result = {int(v): k for k, v in mappings.items()}
            logging.info(f"Loaded {len(result)} author mappings")
            return result
    except FileNotFoundError:
        logging.warning(f"Author mappings not found at {mapping_path}")
        return {}


def visualize_network(G: nx.Graph, author_labels: dict, output_file: str, title: str):
    """Create and save network visualization."""
    logging.info(f"Creating network visualization: {title}")
    logging.info(f"Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    plt.figure(figsize=(20, 20))
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']

    # Process edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    if edge_weights:
        logging.info("Processing edge weights for layout calculation")
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        weight_range = max_weight - min_weight
        logging.info(f"Edge weight range: [{min_weight:.4f}, {max_weight:.4f}]")
    else:
        logging.warning("No edge weights found")

    # Calculate layout with updated parameters
    logging.info("Calculating network layout")
    pos = nx.spring_layout(
        G,
        k=1.5 / np.sqrt(G.number_of_nodes()),
        iterations=100,
        weight='weight',
        seed=42
    )

    # Process edges
    edges = G.edges()
    edge_weights_list = [G[u][v]['weight'] for (u, v) in edges]
    if edge_weights_list:
        max_weight = max(edge_weights_list)
        min_weight = min(edge_weights_list)
        edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights_list]
    else:
        edge_widths = [1]

    # Draw nodes
    logging.info("Drawing network elements")
    node_sizes = [d * 10 for (node, d) in G.degree()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.6)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color=edge_weights_list,
                           edge_cmap=plt.cm.Blues,
                           alpha=0.5)

    # Process labels
    labels = {node: clean_label(author_labels.get(node, str(node))) for node in G.nodes()}
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.02)

    # Draw labels
    nx.draw_networkx_labels(G, pos_attrs, labels,
                            font_size=12,
                            font_weight='bold',
                            bbox=dict(facecolor='white',
                                      edgecolor='none',
                                      alpha=0.7,
                                      pad=0.5))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    logging.info(f"Saving visualization to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Visualization completed")


def save_network_info(G: nx.Graph, author_labels: dict, platform: str, thresholds: dict, output_dir: str):
    """Save network information and author list to a file"""
    logging.info("Preparing network information")

    network_info = {
        "network_statistics": {
            "number_of_nodes": G.number_of_nodes(),
            "number_of_edges": G.number_of_edges(),
            "average_degree": float(sum(dict(G.degree()).values()) / G.number_of_nodes()),
            "density": nx.density(G),
            "number_of_connected_components": nx.number_connected_components(G)
        },
        "threshold_parameters": {
            "similarity_percentile": thresholds['optimal_coordinates']['similarity_percentile'],
            "centrality_percentile": thresholds['optimal_coordinates']['centrality_percentile']
        },
        "authors": {
            str(node): {
                "name": clean_label(author_labels.get(node, f"Unknown-{node}")),
                "degree": G.degree(node),
                "neighbors": len(list(G.neighbors(node)))
            }
            for node in sorted(G.nodes())
        }
    }

    output_file = os.path.join(output_dir, "network_authors.json")
    logging.info(f"Saving network information to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_info, f, indent=2, ensure_ascii=False)

    logging.info("Network information saved successfully")
    return network_info


def create_network_visualization(platform: str):
    """Create and save the network visualization and analysis."""
    logging.info(f"\n{'=' * 50}")
    logging.info(f"Starting network visualization for platform: {platform}")

    # Load data
    try:
        similarity_matrix = load_similarity_matrix(f"../networks/{platform}/author_similarity_matrix.npz")
        author_labels = load_author_mappings(platform)

        output_dir = f"../networks/{platform}/output"
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        threshold_path = os.path.join(output_dir, "thresholds.json")
        logging.info(f"Loading thresholds from {threshold_path}")
        with open(threshold_path, 'r') as f:
            thresholds_data = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

    # Get thresholds
    sim_percentile = thresholds_data['optimal_coordinates']['similarity_percentile']
    cent_percentile = thresholds_data['optimal_coordinates']['centrality_percentile']
    thresholds = thresholds_data['graph_data'][str(cent_percentile)][str(sim_percentile)]

    logging.info(
        f"Using thresholds - Similarity: {thresholds['similarity_threshold']}, Centrality: {thresholds['centrality_threshold']}")

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
    logging.info("Calculating centrality")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
        logging.info("Using eigenvector centrality")
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality failed to converge, using degree centrality")
        eigenvector_centrality = nx.degree_centrality(G_full)

    # Create filtered graph
    logging.info("Creating filtered graph")
    G_filtered = nx.Graph()
    for u, v, w in G_full.edges(data=True):
        if np.round(w['weight'], decimals=2) >= thresholds['similarity_threshold']:
            if (eigenvector_centrality[u] >= thresholds['centrality_threshold'] and
                    eigenvector_centrality[v] >= thresholds['centrality_threshold']):
                G_filtered.add_edge(u, v, weight=w['weight'])

    logging.info(f"Filtered graph has {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges")

    # Create visualization
    visualize_network(
        G_filtered,
        author_labels,
        os.path.join(output_dir, "max_increment_network.png"),
        f"{platform.capitalize()} Network at Maximum Increment\n(Sim: {sim_percentile}, Cent: {cent_percentile})"
    )

    # Save and display network information
    network_info = save_network_info(G_filtered, author_labels, platform, thresholds_data, output_dir)

    logging.info("\nNetwork Statistics Summary:")
    logging.info(f"Number of nodes: {network_info['network_statistics']['number_of_nodes']}")
    logging.info(f"Number of edges: {network_info['network_statistics']['number_of_edges']}")
    logging.info(f"Average degree: {network_info['network_statistics']['average_degree']:.2f}")
    logging.info(f"Network density: {network_info['network_statistics']['density']:.3f}")
    logging.info(
        f"Number of connected components: {network_info['network_statistics']['number_of_connected_components']}")
    logging.info(f"\nResults saved to: {output_dir}")
    logging.info("Network visualization completed successfully")
