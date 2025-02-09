import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import json
import re
from typing import Dict, Tuple
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_label(text: str) -> str:
    """Clean label text by removing emojis and problematic Unicode characters."""
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


def get_platform_colors() -> Dict[str, str]:
    """Define colors for each platform."""
    return {
        'telegram': '#FF0000',  # Telegram red
        'gab': '#00FF00',  # Gab green
        'vk': '#FFFF00',  # VK yellow
        'fediverse': '#0000FF',  # Fediverse blue
        'minds': '#00FFFF', # Minds cyan
        'unknown': '#000000'  # Unknown black
    }


def load_similarity_matrix(file_path: str) -> np.ndarray:
    """Load the similarity matrix from a file."""
    logging.info(f"Loading similarity matrix from {file_path}")
    data = np.load(file_path)
    matrix = csr_matrix((data['data'], data['indices'], data['indptr']),
                        shape=data['shape']).toarray()
    matrix = np.round(matrix, decimals=2)
    logging.info(f"Matrix shape: {matrix.shape}")
    logging.info(f"Non-zero elements: {np.count_nonzero(matrix)}")
    return matrix


def load_author_data(base_dir: str) -> Tuple[Dict, Dict]:
    """Load author mappings and platform information."""
    logging.info("Loading author mappings...")
    with open(os.path.join(base_dir, "author_mappings.json"), 'r') as f:
        mapping_data = json.load(f)
        author_to_idx = mapping_data['author_to_idx']
        idx_to_author = {int(v): k for k, v in author_to_idx.items()}
        author_platforms = mapping_data.get('author_platforms', {})

    logging.info(f"Loaded {len(idx_to_author)} authors with platform information")
    return idx_to_author, author_platforms


def visualize_cross_platform_network(G: nx.Graph,
                                     author_labels: dict,
                                     author_platforms: dict,
                                     output_file: str,
                                     title: str):
    """Create and save the network visualization with detailed centrality logging."""
    logging.info("\n=== Creating Network Visualization ===")
    logging.info(f"Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Calculate eigenvector centrality on the filtered graph
    logging.info("\nCalculating eigenvector centrality on filtered graph...")
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')

        # Create a list of nodes with their properties for sorting and logging
        node_info = []
        for node in G.nodes():
            degree = G.degree(node)
            weighted_degree = sum(G[node][n]['weight'] for n in G.neighbors(node))
            author_name = clean_label(author_labels.get(node, f"Unknown-{node}"))
            platform = author_platforms.get(author_labels.get(node, ''), 'Unknown')

            node_info.append({
                'node': node,
                'centrality': centrality[node],
                'degree': degree,
                'weighted_degree': weighted_degree,
                'author': author_name,
                'platform': platform
            })

        # Sort nodes by centrality value
        sorted_nodes = sorted(node_info, key=lambda x: x['centrality'], reverse=True)

        # Log all nodes with their centrality values and other properties
        logging.info("\nEigenvector Centrality Values for All Nodes (sorted by centrality):")
        logging.info("\nFormat: NodeID: Centrality | Degree | Weighted Degree | Platform | Author")
        logging.info("-" * 80)

        for idx, info in enumerate(sorted_nodes, 1):
            logging.info(
                f"{idx}. Node {info['node']}: {info['centrality']:.6f} | "
                f"Degree: {info['degree']} | "
                f"Weighted Degree: {info['weighted_degree']:.4f} | "
                f"Platform: {info['platform']} | "
                f"Author: {info['author']}"
            )

        logging.info("-" * 80)

        # Calculate and log some statistics
        centrality_values = [info['centrality'] for info in sorted_nodes]
        mean_centrality = np.mean(centrality_values)
        median_centrality = np.median(centrality_values)
        std_centrality = np.std(centrality_values)

        logging.info("\nCentrality Statistics:")
        logging.info(f"Mean Centrality: {mean_centrality:.6f}")
        logging.info(f"Median Centrality: {median_centrality:.6f}")
        logging.info(f"Standard Deviation: {std_centrality:.6f}")
        logging.info(f"Maximum Centrality: {max(centrality_values):.6f}")
        logging.info(f"Minimum Centrality: {min(centrality_values):.6f}")

    except (nx.PowerIterationFailedConvergence, ValueError) as e:
        logging.error(f"Error calculating eigenvector centrality: {str(e)}")
        return

    # Get top nodes by this new centrality calculation
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_node_ids = [node for node, _ in top_nodes]

    # Create figure
    plt.figure(figsize=(20, 20))
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']

    # Calculate layout
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
    platform_counts = {}
    for node in G.nodes():
        platform = author_platforms.get(author_labels.get(node, ''), '').lower()
        if not platform:
            platform = 'unknown'
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
        node_colors.append(platform_colors.get(platform, platform_colors['unknown']))

    # Draw edges
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for (u, v) in edges]

    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight)
                       for w in edge_weights]
    else:
        edge_widths = [1]

    # Draw the base network
    node_sizes = [centrality[node] * 5000 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.7)

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color='gray',
                           alpha=0.3)

    # Create labels for top nodes
    labels = {node: str(node) for node in top_node_ids}

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels,
                            font_size=8,
                            font_weight='bold',
                            bbox=dict(facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.8,
                                      pad=4))

    # Create legends
    unique_platforms = set(platform_counts.keys())
    platform_legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=platform_colors.get(p.lower(), platform_colors['unknown']),
                                           markersize=10, label=p.upper())
                                for p in sorted(unique_platforms)]

    # Create mapping text for top nodes
    mapping_text = "Top 5 Nodes by Eigenvector Centrality:\n\n"
    for node, cent_value in top_nodes:
        author_name = clean_label(author_labels.get(node, f"Unknown-{node}"))
        platform = author_platforms.get(author_labels.get(node, ''), 'Unknown')
        degree = G.degree(node)
        weighted_degree = sum(G[node][n]['weight'] for n in G.neighbors(node))

        mapping_text += f"{node}: {author_name}\n"
        mapping_text += f"Platform: {platform}\n"
        mapping_text += f"Centrality: {cent_value:.4f}\n"
        mapping_text += f"Degree: {degree}\n"
        mapping_text += f"Weighted Degree: {weighted_degree:.4f}\n\n"

    # Add legends to plot
    plt.legend(handles=platform_legend_elements,
               loc='upper left',
               title="Platforms",
               bbox_to_anchor=(0.02, 0.98))

    plt.text(0.98, 0.98, mapping_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=10,
             fontfamily='monospace',
             bbox=dict(facecolor='white',
                       edgecolor='gray',
                       alpha=0.8,
                       pad=10))

    plt.title(title, pad=20)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Visualization completed successfully")



def save_network_info(G: nx.Graph,
                      author_labels: dict,
                      author_platforms: dict,
                      thresholds: dict,
                      output_dir: str) -> Dict:
    """Save network information and author list to a file."""
    logging.info("\n=== Saving Network Information ===")

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
    logging.info(f"Saving network information to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_info, f, indent=2, ensure_ascii=False)

    logging.info("Network information saved successfully")
    return network_info


def create_network_visualization():
    """Create and save the cross-platform network visualization."""
    logging.info("\n=== Starting Cross-Platform Network Visualization ===")

    # Set up paths
    base_dir = "../networks/cross_platform"
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Working directory: {base_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Load data
    similarity_matrix = load_similarity_matrix(os.path.join(base_dir, "author_similarity_matrix.npz"))
    author_labels, author_platforms = load_author_data(base_dir)

    logging.info("Loading threshold data...")
    with open(os.path.join(output_dir, "thresholds.json"), 'r') as f:
        thresholds_data = json.load(f)

    # Get optimal coordinates
    sim_percentile = thresholds_data['optimal_coordinates']['similarity_percentile']
    cent_percentile = thresholds_data['optimal_coordinates']['centrality_percentile']
    logging.info(f"Using similarity percentile: {sim_percentile}")
    logging.info(f"Using centrality percentile: {cent_percentile}")

    # Get thresholds
    thresholds = thresholds_data['graph_data'][str(cent_percentile)][str(sim_percentile)]
    similarity_threshold = thresholds['similarity_threshold']
    centrality_threshold = thresholds['centrality_threshold']
    logging.info(f"Similarity threshold: {similarity_threshold}")
    logging.info(f"Centrality threshold: {centrality_threshold}")

    # Create full graph
    logging.info("Creating initial graph...")
    num_nodes = similarity_matrix.shape[0]
    G_full = nx.Graph()

    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > 0:
                G_full.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_count += 1

    logging.info(f"Initial graph created with {num_nodes} nodes and {edge_count} edges")

    # Calculate centrality
    logging.info("Calculating centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_full, max_iter=1000, weight='weight')
        logging.info("Using eigenvector centrality")
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality failed to converge, using degree centrality instead")
        eigenvector_centrality = nx.degree_centrality(G_full)

    # Create filtered graph
    logging.info("Creating filtered graph...")
    G_filtered = nx.Graph()

    for u, v, w in G_full.edges(data=True):
        if np.round(w['weight'], decimals=2) >= similarity_threshold:
            if (eigenvector_centrality[u] >= centrality_threshold and
                    eigenvector_centrality[v] >= centrality_threshold):
                G_filtered.add_edge(u, v, weight=w['weight'])

    logging.info(
        f"Filtered graph created with {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges")

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
    logging.info("\n=== Network Analysis Summary ===")
    logging.info(f"Number of nodes: {network_info['network_statistics']['number_of_nodes']}")
    logging.info(f"Number of edges: {network_info['network_statistics']['number_of_edges']}")
    logging.info(f"Average degree: {network_info['network_statistics']['average_degree']:.2f}")
    logging.info(f"Network density: {network_info['network_statistics']['density']:.3f}")
    logging.info("\nPlatform Distribution:")
    for platform, count in network_info['platform_statistics'].items():
        logging.info(f"{platform}: {count} nodes")
    logging.info(f"\nNetwork information saved to: {os.path.join(output_dir, 'network_authors.json')}")
    logging.info("\nVisualization process completed successfully")


if __name__ == "__main__":
    create_network_visualization()