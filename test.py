from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set, TypeAlias, NamedTuple, Callable
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, spmatrix, csr_matrix
from tqdm import tqdm
import re
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor

# Type aliases
NodeID: TypeAlias = str
GroupID: TypeAlias = int
Weight: TypeAlias = float
Metrics: TypeAlias = Dict[str, any]
CleanerFn: TypeAlias = Callable[[str], str]


class NetworkData(NamedTuple):
    """Immutable container for network data."""
    similarity_matrix: spmatrix
    author_mappings: Dict[str, Dict[str, int]]
    similar_groups: List[List[str]]


class NetworkPaths(NamedTuple):
    """Immutable container for file paths."""
    matrix_path: Path
    mappings_path: Path
    groups_path: Path
    metrics_path: Path
    viz_path: Path


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable configuration for network analysis."""
    min_edge_weight: float = 0.1
    max_label_length: int = 20
    fig_size: Tuple[int, int] = (15, 10)
    node_size: int = 1000
    alpha: float = 0.7
    font_size: int = 8
    edge_color: str = '#CBD5E0'
    node_colors: Tuple[str, str, str] = ('#4299E1', '#48BB78', '#A0AEC0')


def get_paths(data_dir: str, output_dir: str) -> NetworkPaths:
    """Pure function to create file paths."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return NetworkPaths(
        matrix_path=data_path / 'author_similarity_matrix.npz',
        mappings_path=data_path / 'author_url_mappings.json',
        groups_path=data_path / 'similar_author_groups.json',
        metrics_path=output_path / 'network_metrics.json',
        viz_path=output_path / 'network_visualization.png'
    )


def create_label_cleaner(max_length: int) -> CleanerFn:
    """Creates a pure function for cleaning labels."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"❌🔊"
        u"]+"
    )

    @lru_cache(maxsize=1024)
    def clean_label(text: str) -> str:
        if not isinstance(text, str):
            return str(text)

        # Remove emojis and special characters
        text = emoji_pattern.sub('', text)
        # Keep only alphanumeric and basic punctuation
        text = ''.join(c for c in text if c.isalnum() or c in ' .-')
        text = text.strip()

        return f"{text[:max_length]}..." if len(text) > max_length else text

    return clean_label


def load_network_data(paths: NetworkPaths) -> NetworkData:
    """Pure function to load network data."""
    if not all(p.exists() for p in [paths.matrix_path, paths.mappings_path, paths.groups_path]):
        raise FileNotFoundError("Required network data files not found")

    with ThreadPoolExecutor() as executor:
        matrix_future = executor.submit(load_npz, str(paths.matrix_path))
        mappings_future = executor.submit(lambda: json.loads(
            paths.mappings_path.read_text(encoding='utf-8')))
        groups_future = executor.submit(lambda: json.loads(
            paths.groups_path.read_text(encoding='utf-8')))

    return NetworkData(
        similarity_matrix=matrix_future.result(),
        author_mappings=mappings_future.result(),
        similar_groups=groups_future.result()
    )


def find_author_groups(similar_groups: List[List[str]]) -> Dict[str, int]:
    """Pure function to create author-to-group mapping."""
    return {
        author: group_idx
        for group_idx, group in enumerate(similar_groups)
        for author in group
    }


def create_network(
        data: NetworkData,
        min_edge_weight: float
) -> nx.Graph:
    """Pure function to create network graph."""
    G = nx.Graph()
    authors = list(data.author_mappings['author_to_idx'].keys())

    # Create author-to-group mapping
    author_groups = find_author_groups(data.similar_groups)

    # Add nodes
    G.add_nodes_from(
        (author, {'group': author_groups.get(author, -1)})
        for author in authors
    )

    # Add edges
    matrix_coo = data.similarity_matrix.tocoo()
    edges = [
        (authors[i], authors[j], {'weight': v})
        for i, j, v in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data)
        if i < j and v >= min_edge_weight
    ]
    G.add_edges_from(edges)

    return G


def calculate_metrics(G: nx.Graph, clean_label: CleanerFn) -> Metrics:
    """Pure function to calculate network metrics."""
    with ThreadPoolExecutor() as executor:
        degree_future = executor.submit(nx.degree_centrality, G)
        betweenness_future = executor.submit(nx.betweenness_centrality, G)

        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'degree_centrality': {
                clean_label(k): v
                for k, v in degree_future.result().items()
            },
            'betweenness_centrality': {
                clean_label(k): v
                for k, v in betweenness_future.result().items()
            }
        }

        try:
            import community
            metrics['communities'] = {
                clean_label(k): v
                for k, v in community.best_partition(G).items()
            }
        except ImportError:
            print("Note: Install python-louvain package for community detection")

        return metrics


def create_visualization(
        G: nx.Graph,
        config: NetworkConfig,
        clean_label: CleanerFn
) -> plt.Figure:
    """Pure function to create network visualization."""
    fig = plt.figure(figsize=config.fig_size)

    # Prepare visualization data
    node_colors = [
        config.node_colors[G.nodes[node]['group'] if G.nodes[node]['group'] in (0, 1) else 2]
        for node in G.nodes()
    ]

    pos = nx.spring_layout(G, k=1, iterations=50)
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

    # Set up plot
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

    # Draw network
    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=config.node_size,
        alpha=config.alpha,
        with_labels=True,
        labels={node: clean_label(node) for node in G.nodes()},
        font_size=config.font_size,
        edge_color=config.edge_color,
        width=edge_widths
    )

    plt.title('Author Network Analysis', pad=20)
    return fig


def save_outputs(
        metrics: Metrics,
        fig: plt.Figure,
        paths: NetworkPaths
) -> None:
    """Function to save analysis outputs."""
    # Save metrics
    paths.metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding='utf-8'
    )
    print(f"Saved network metrics to {paths.metrics_path}")

    # Save visualization
    fig.savefig(paths.viz_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {paths.viz_path}")


def analyze_network(data_dir: str, output_dir: str) -> None:
    """Main pipeline function composing all operations."""
    try:
        # Initialize configuration
        config = NetworkConfig()
        paths = get_paths(data_dir, output_dir)
        clean_label = create_label_cleaner(config.max_label_length)

        print("Loading network data...")
        data = load_network_data(paths)

        print("Constructing network...")
        G = create_network(data, config.min_edge_weight)

        print("Analyzing network...")
        metrics = calculate_metrics(G, clean_label)

        print("Generating visualization...")
        fig = create_visualization(G, config, clean_label)

        save_outputs(metrics, fig, paths)

    except Exception as e:
        print(f"Error during network analysis: {str(e)}")
        raise


def main() -> None:
    """Entry point."""
    data_dir = "./networks/gettr"
    output_dir = "./output"
    analyze_network(data_dir, output_dir)


if __name__ == "__main__":
    main()