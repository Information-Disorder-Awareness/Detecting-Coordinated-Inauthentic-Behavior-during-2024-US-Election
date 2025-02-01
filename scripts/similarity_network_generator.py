from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypeAlias, NamedTuple
from functools import partial
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from custom_types.Platform import Platform
import re
import json
from tqdm import tqdm

# Type aliases
AuthorUrlPair: TypeAlias = Tuple[str, str]
AuthorIdx: TypeAlias = Dict[str, int]
UrlIdx: TypeAlias = Dict[str, int]
SimilarityMatrix: TypeAlias = csr_matrix
AuthorGroups: TypeAlias = List[List[str]]


class NetworkPaths(NamedTuple):
    """Immutable paths for network analysis."""
    input_file: Path
    groups_file: Path
    matrix_file: Path
    mappings_file: Path


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for similarity network analysis."""
    threshold: float = 0.5
    batch_size: int = 10000
    url_pattern: str = r'https?://[^\s<>"]+|www\.[^\s<>"]+|bit\.ly/[^\s<>"]+|t\.co/[^\s<>"]+|tinyurl\.com/[^\s<>"]+|is\.gd/[^\s<>"]+'


def get_network_paths(platform: Platform) -> NetworkPaths:
    """Create network paths from platform. Pure function with no side effects."""
    platform_dir = Path(f"./datasets/{platform}")
    network_dir = Path(f"./networks/{platform}")

    return NetworkPaths(
        input_file=platform_dir / "filtered_posts_cleaned.csv",
        groups_file=network_dir / "similar_author_groups.json",
        matrix_file=network_dir / "author_similarity_matrix.npz",
        mappings_file=network_dir / "author_url_mappings.json"
    )


def ensure_directories(paths: NetworkPaths) -> None:
    """Ensure required directories exist."""
    paths.groups_file.parent.mkdir(parents=True, exist_ok=True)


def extract_urls(text: str, pattern: str) -> Set[str]:
    """Extract and normalize URLs from text. Pure function."""
    if not isinstance(text, str):
        return set()

    urls = re.findall(pattern, text, re.IGNORECASE)
    return {url.lower().rstrip('/.').replace('www.', '') for url in urls}


def process_batch(batch: pd.DataFrame, config: NetworkConfig) -> List[AuthorUrlPair]:
    """Process a batch of data. Pure function."""
    url_extractor = partial(extract_urls, pattern=config.url_pattern)
    return [
        (row['author'], url)
        for _, row in batch.iterrows()
        for url in url_extractor(row['text'])
    ]


def process_data(paths: NetworkPaths, config: NetworkConfig) -> List[AuthorUrlPair]:
    """Process input data. Returns author-URL pairs."""
    if not paths.input_file.exists():
        raise FileNotFoundError(f"Required input file not found: {paths.input_file}")

    print(f"Reading data from {paths.input_file}")
    df = pd.read_csv(paths.input_file)
    print(f"Loaded {len(df)} posts from dataset")

    pairs: List[AuthorUrlPair] = []
    total_batches = (len(df) + config.batch_size - 1) // config.batch_size

    for start_idx in tqdm(range(0, len(df), config.batch_size), total=total_batches, desc="Processing posts"):
        batch = df.iloc[start_idx:start_idx + config.batch_size]
        pairs.extend(process_batch(batch, config))

    print(f"Extracted {len(pairs)} author-URL pairs")
    return pairs


def create_mappings(pairs: List[AuthorUrlPair]) -> Tuple[AuthorIdx, UrlIdx]:
    """Create index mappings. Pure function."""
    unique_authors = sorted({author for author, _ in pairs})
    unique_urls = sorted({url for _, url in pairs})

    print(f"Found {len(unique_authors)} unique authors and {len(unique_urls)} unique URLs")

    return (
        {author: idx for idx, author in enumerate(unique_authors)},
        {url: idx for idx, url in enumerate(unique_urls)}
    )


def build_frequency_matrix(
        pairs: List[AuthorUrlPair],
        author_to_idx: AuthorIdx,
        url_to_idx: UrlIdx
) -> SimilarityMatrix:
    """Build frequency matrix. Pure function."""
    print("Building frequency matrix...")

    rows = np.fromiter((author_to_idx[author] for author, _ in pairs), dtype=np.int32)
    cols = np.fromiter((url_to_idx[url] for _, url in pairs), dtype=np.int32)
    data = np.ones(len(pairs), dtype=np.float32)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(author_to_idx), len(url_to_idx))
    )

    print(f"Created sparse matrix with shape {matrix.shape}")
    return matrix


def calculate_similarity(matrix: SimilarityMatrix) -> Tuple[SimilarityMatrix, SimilarityMatrix]:
    """Calculate similarity matrices. Pure function."""
    print("Calculating TF-IDF transformation...")
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf_matrix = transformer.fit_transform(matrix)

    print("Computing author similarity matrix...")
    normalized_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    author_similarity = normalized_matrix @ normalized_matrix.T

    return author_similarity, tfidf_matrix


def find_similar_authors(
        similarity_matrix: SimilarityMatrix,
        author_to_idx: AuthorIdx,
        threshold: float
) -> AuthorGroups:
    """Find similar author groups. Pure function."""
    print(f"Finding similar authors with threshold {threshold}")

    authors = list(author_to_idx.keys())
    sim_matrix = similarity_matrix.toarray()
    similar_indices = (sim_matrix >= threshold).nonzero()

    similar_authors = {
        tuple(sorted([authors[i], authors[j]]))
        for i, j in tqdm(zip(*similar_indices), desc="Finding similar authors")
        if i < j  # Process upper triangle only
    }

    return [list(group) for group in similar_authors]


def save_results(
        paths: NetworkPaths,
        groups: AuthorGroups,
        similarity: SimilarityMatrix,
        author_to_idx: AuthorIdx,
        url_to_idx: UrlIdx
) -> None:
    """Save analysis results."""
    with open(paths.groups_file, 'w') as f:
        json.dump(groups, f, indent=2)

    save_npz(str(paths.matrix_file), similarity)

    with open(paths.mappings_file, 'w') as f:
        json.dump({
            'author_to_idx': author_to_idx,
            'url_to_idx': url_to_idx
        }, f, indent=2)

    print("Successfully saved all results")


def analyze_url_similarity_network(platform: Platform) -> None:
    """Pipeline-compatible main function."""
    try:
        paths = get_network_paths(platform)
        config = NetworkConfig()

        # Early return if analysis is already done
        if paths.groups_file.exists() and paths.matrix_file.exists():
            print(f"URL similarity analysis already completed for {platform}")
            return

        print(f"Starting URL similarity analysis for {platform}...")
        ensure_directories(paths)

        # Execute analysis pipeline using pure functions
        pairs = process_data(paths, config)
        author_to_idx, url_to_idx = create_mappings(pairs)
        frequency_matrix = build_frequency_matrix(pairs, author_to_idx, url_to_idx)
        similarity_matrix, _ = calculate_similarity(frequency_matrix)
        similar_groups = find_similar_authors(similarity_matrix, author_to_idx, config.threshold)

        save_results(paths, similar_groups, similarity_matrix, author_to_idx, url_to_idx)
        print(f"Analysis completed for {platform}")

    except Exception as e:
        print(f"Error during similarity network analysis: {str(e)}")
        raise


if __name__ == "__main__":
    analyze_url_similarity_network(Platform.TELEGRAM)