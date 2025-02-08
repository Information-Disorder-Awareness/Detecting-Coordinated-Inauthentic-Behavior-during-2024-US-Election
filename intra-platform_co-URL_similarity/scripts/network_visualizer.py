import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
import re


def clean_label(text):
    """Clean label text by removing emojis and problematic Unicode characters"""
    # Pattern to match emojis and other special characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Remove emojis and special characters
    cleaned_text = emoji_pattern.sub('', text)
    # Remove any remaining non-ASCII characters except common punctuation
    cleaned_text = ''.join(c for c in cleaned_text if c.isascii() or c.isspace())
    return cleaned_text.strip()


def load_similarity_matrix(file_path: str) -> np.ndarray:
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    return np.round(matrix, decimals=2)


def load_author_mappings(platform: str) -> dict:
    """Load author mappings from the JSON file"""
    mapping_path = f"../networks/{platform}/author_url_mappings.json"
    with open(mapping_path, 'r') as f:
        data = json.load(f)
        # The mappings are stored under 'author_to_idx' key
        mappings = data['author_to_idx']
        # Invert the mappings
        return {int(v): k for k, v in mappings.items()}


def visualize_network(G: nx.Graph, author_labels: dict, output_file: str, title: str):
    plt.figure(figsize=(20, 20))

    # Set a font that handles Unicode better
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']

    # Convert weights to distances (higher weight = shorter distance)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    # Scale weights to be between 0 and 1
    if edge_weights:
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        weight_range = max_weight - min_weight

        # Create a dictionary of edge distances (inverse of weights)
        distances = {(u, v): 1 - (w - min_weight) / weight_range
                     for (u, v), w in edge_weights.items()}
    else:
        distances = None

    # Use spring layout with the distance parameter
    pos = nx.spring_layout(
        G,
        k=1.5 / np.sqrt(G.number_of_nodes()),  # Increased k for more spread
        iterations=100,  # More iterations for better convergence
        weight='weight',  # Use weight attribute
        seed=42  # For reproducibility
    )

    # Draw edges with width and color based on weight
    edges = G.edges()
    edge_weights_list = [G[u][v]['weight'] for (u, v) in edges]

    # Normalize edge widths
    if edge_weights_list:
        max_weight = max(edge_weights_list)
        min_weight = min(edge_weights_list)
        edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights_list]
    else:
        edge_widths = [1]

    # Draw nodes with sizes based on degree
    node_sizes = [d * 10 for (node, d) in G.degree()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.6)

    # Draw edges with varying width and color intensity based on weight
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color=edge_weights_list,
                           edge_cmap=plt.cm.Blues,
                           alpha=0.5)

    # Add labels with cleaned author names
    labels = {node: clean_label(author_labels.get(node, str(node))) for node in G.nodes()}

    # Adjust label positions slightly above nodes
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.02)

    # Draw labels with larger font size and better visibility
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def save_network_info(G: nx.Graph, author_labels: dict, platform: str, thresholds: dict, output_dir: str):
    """Save network information and author list to a file"""
    # Prepare network information
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

    # Save to file
    output_file = os.path.join(output_dir, "network_authors.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_info, f, indent=2, ensure_ascii=False)

    return network_info


def create_network_visualization(platform: str):
    # Load similarity matrix
    similarity_matrix = load_similarity_matrix(f"../networks/{platform}/author_similarity_matrix.npz")

    # Load author mappings
    try:
        author_labels = load_author_mappings(platform)
    except FileNotFoundError:
        print(f"Warning: Author mappings not found for {platform}. Using numeric indices as labels.")
        author_labels = {}

    # Load thresholds and optimal coordinates
    output_dir = f"../networks/{platform}/output"
    with open(os.path.join(output_dir, "thresholds.json"), 'r') as f:
        thresholds_data = json.load(f)

    # Get optimal coordinates
    sim_percentile = thresholds_data['optimal_coordinates']['similarity_percentile']
    cent_percentile = thresholds_data['optimal_coordinates']['centrality_percentile']

    # Get corresponding thresholds
    thresholds = thresholds_data['graph_data'][str(cent_percentile)][str(sim_percentile)]
    similarity_threshold = thresholds['similarity_threshold']
    centrality_threshold = thresholds['centrality_threshold']

    # Create full graph
    num_nodes = similarity_matrix.shape[0]
    G_full = nx.Graph()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > 0:
                G_full.add_edge(i, j, weight=similarity_matrix[i, j])

    # Calculate centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = nx.degree_centrality(G_full)

    # Create filtered graph
    G_filtered = nx.Graph()

    for u, v, w in G_full.edges(data=True):
        if np.round(w['weight'], decimals=2) >= similarity_threshold:
            if (eigenvector_centrality[u] >= centrality_threshold and
                    eigenvector_centrality[v] >= centrality_threshold):
                G_filtered.add_edge(u, v, weight=w['weight'])

    # Visualize the network with author labels
    visualize_network(
        G_filtered,
        author_labels,
        os.path.join(output_dir, "max_increment_network.png"),
        f"{platform.capitalize()} Network at Maximum Increment\n(Sim: {sim_percentile}, Cent: {cent_percentile})"
    )

    # Save network information and author list
    network_info = save_network_info(G_filtered, author_labels, platform, thresholds_data, output_dir)

    # Print summary to console
    print(f"\nNetwork Statistics:")
    print(f"Number of nodes: {network_info['network_statistics']['number_of_nodes']}")
    print(f"Number of edges: {network_info['network_statistics']['number_of_edges']}")
    print(f"Average degree: {network_info['network_statistics']['average_degree']:.2f}")
    print(f"Network density: {network_info['network_statistics']['density']:.3f}")
    print(f"Number of connected components: {network_info['network_statistics']['number_of_connected_components']}")
    print(f"\nNetwork information saved to: {os.path.join(output_dir, 'network_authors.json')}")


if __name__ == "__main__":
    create_network_visualization("telegram")