import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
import re
from typing import Dict, Tuple


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
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    return np.round(matrix, decimals=2)


def load_author_data(base_dir: str) -> Tuple[Dict, Dict]:
    """Load author mappings and platform information"""
    # Load author mappings
    with open(os.path.join(base_dir, "author_mappings.json"), 'r') as f:
        mapping_data = json.load(f)
        author_to_idx = mapping_data['author_to_idx']
        idx_to_author = {int(v): k for k, v in author_to_idx.items()}

    # Load author platform information from similar authors file
    with open(os.path.join(base_dir, "author_similar_authors.json"), 'r') as f:
        similar_authors = json.load(f)

    # Create platform mapping
    author_platforms = {}
    for pair in similar_authors:
        author_platforms[pair['author1']] = pair['platform1']
        author_platforms[pair['author2']] = pair['platform2']

    return idx_to_author, author_platforms


def get_platform_colors() -> Dict[str, str]:
    """Define colors for each platform"""
    return {
        'telegram': '#0088cc',  # Telegram blue
        'gab': '#00ff00',  # Gab green
        'gettr': '#ff0000',  # Gettr red
        'vk': '#4C75A3',  # VK blue
        'fediverse': '#6364FF'  # Fediverse purple/blue
    }


def visualize_cross_platform_network(G: nx.Graph,
                                     author_labels: dict,
                                     author_platforms: dict,
                                     output_file: str,
                                     title: str):
    plt.figure(figsize=(20, 20))
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']

    # Get position layout
    pos = nx.spring_layout(
        G,
        k=1.5 / np.sqrt(G.number_of_nodes()),
        iterations=100,
        weight='weight',
        seed=42
    )

    # Set up platform colors
    platform_colors = get_platform_colors()

    # Create node colors list based on platforms
    node_colors = []
    for node in G.nodes():
        platform = author_platforms.get(author_labels.get(node, ''), '').lower()
        if platform in platform_colors:
            node_colors.append(platform_colors[platform])
        else:
            print(f"Warning: Unknown platform '{platform}' for author {author_labels.get(node, node)}")

    # Draw edges with width based on weight
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for (u, v) in edges]

    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight)
                       for w in edge_weights]
    else:
        edge_widths = [1]

    # Draw nodes with platform-based colors
    node_sizes = [d * 10 for (node, d) in G.degree()]
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.7)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color='gray',
                           alpha=0.3)

    # Add labels
    labels = {
        node: f"{clean_label(author_labels.get(node, str(node)))}\n({author_platforms.get(author_labels.get(node, ''), 'Unknown')})"
        for node in G.nodes()}

    # Adjust label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.02)

    # Draw labels
    nx.draw_networkx_labels(G, pos_attrs, labels,
                            font_size=10,
                            font_weight='bold',
                            bbox=dict(facecolor='white',
                                      edgecolor='none',
                                      alpha=0.7,
                                      pad=0.5))

    # Add legend
    unique_platforms = set(author_platforms.values())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=platform_colors.get(p.lower(), platform_colors['default']),
                                  markersize=10, label=p)
                       for p in sorted(unique_platforms)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def save_network_info(G: nx.Graph,
                      author_labels: dict,
                      author_platforms: dict,
                      thresholds: dict,
                      output_dir: str):
    """Save network information and author list to a file"""
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
                "platform": author_platforms.get(author_labels.get(node, ''), 'Unknown'),
                "degree": G.degree(node),
                "neighbors": len(list(G.neighbors(node)))
            }
            for node in sorted(G.nodes())
        }
    }

    # Add platform statistics
    platform_counts = {}
    for node in G.nodes():
        platform = author_platforms.get(author_labels.get(node, ''), 'Unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1

    network_info["platform_statistics"] = platform_counts

    output_file = os.path.join(output_dir, "network_authors.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_info, f, indent=2, ensure_ascii=False)

    return network_info


def create_network_visualization():
    # Set up paths
    base_dir = "../networks/cross_platform"
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    similarity_matrix = load_similarity_matrix(os.path.join(base_dir, "author_similarity_matrix.npz"))
    author_labels, author_platforms = load_author_data(base_dir)

    with open(os.path.join(output_dir, "thresholds.json"), 'r') as f:
        thresholds_data = json.load(f)

    # Get optimal coordinates
    sim_percentile = thresholds_data['optimal_coordinates']['similarity_percentile']
    cent_percentile = thresholds_data['optimal_coordinates']['centrality_percentile']

    # Get thresholds
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

    # Visualize the network
    visualize_cross_platform_network(
        G_filtered,
        author_labels,
        author_platforms,
        os.path.join(output_dir, "cross_platform_network.png"),
        f"Cross-Platform Network at Maximum Increment\n(Sim: {sim_percentile}, Cent: {cent_percentile})"
    )

    # Save network information
    network_info = save_network_info(G_filtered, author_labels, author_platforms, thresholds_data, output_dir)

    # Print summary
    print(f"\nNetwork Statistics:")
    print(f"Number of nodes: {network_info['network_statistics']['number_of_nodes']}")
    print(f"Number of edges: {network_info['network_statistics']['number_of_edges']}")
    print(f"Average degree: {network_info['network_statistics']['average_degree']:.2f}")
    print(f"Network density: {network_info['network_statistics']['density']:.3f}")
    print(f"\nPlatform Distribution:")
    for platform, count in network_info['platform_statistics'].items():
        print(f"{platform}: {count} nodes")
    print(f"\nNetwork information saved to: {os.path.join(output_dir, 'network_authors.json')}")


if __name__ == "__main__":
    create_network_visualization()