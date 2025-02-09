from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import re
import json
from tqdm import tqdm
from datetime import timedelta
from typing import List, Set, Tuple, Dict
import logging

from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(frozen=True)
class NetworkConfig:
    threshold: float = 0.5
    batch_size: int = 10000
    timeframe_days: int = 7
    url_pattern: str = r'https?://[^\s<>"]+|www\.[^\s<>"]+|bit\.ly/[^\s<>"]+|t\.co/[^\s<>"]+|tinyurl\.com/[^\s<>"]+|is\.gd/[^\s<>"]+'
    platforms: List[str] = (Platform.FEDIVERSE, Platform.TELEGRAM, Platform.GAB, Platform.VK, Platform.MINDS)


def extract_urls(text: str, pattern: str) -> Set[str]:
    """Extract and normalize URLs from text."""
    if not isinstance(text, str):
        return set()
    urls = re.findall(pattern, text, re.IGNORECASE)
    return {url.lower().rstrip('/.').replace('www.', '') for url in urls}


def create_frequency_matrix(pairs: List[Tuple[str, str]],
                            author_to_idx: Dict[str, int],
                            url_to_idx: Dict[str, int]) -> csr_matrix:
    """Create a sparse frequency matrix from author-URL pairs."""
    rows = [author_to_idx[author] for author, _ in pairs]
    cols = [url_to_idx[url] for _, url in pairs]
    return csr_matrix(
        (np.ones(len(pairs)), (rows, cols)),
        shape=(len(author_to_idx), len(url_to_idx))
    )


def calculate_similarity_matrix(freq_matrix: csr_matrix) -> csr_matrix:
    """Calculate similarity matrix using TF-IDF and cosine similarity."""
    tfidf_matrix = TfidfTransformer(smooth_idf=True).fit_transform(freq_matrix)
    normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    return normalized @ normalized.T


def load_platform_data(platform: str, base_dir: Path) -> pd.DataFrame:
    """Load and prepare data for a specific platform."""
    input_file = base_dir / "datasets" / platform / "filtered_posts_cleaned.csv"
    logging.info(f"Loading data from {input_file}")

    df = pd.read_csv(input_file)
    df['platform'] = platform
    df['create_time'] = pd.to_datetime(df['create_time'])

    logging.info(f"Loaded {len(df)} records from {platform}")
    return df


def process_timeframe_data(timeframe_df: pd.DataFrame,
                           config: NetworkConfig) -> List[Tuple[str, str]]:
    """Process data for a specific timeframe and extract URL pairs."""
    timeframe_pairs = []
    for _, row in timeframe_df.iterrows():
        urls = extract_urls(row['text'], config.url_pattern)
        timeframe_pairs.extend((row['author'], url) for url in urls)
    return timeframe_pairs


def find_cross_platform_pairs(average_similarity: np.ndarray,
                              authors: List[str],
                              author_platform_dict: Dict[str, str],
                              threshold: float) -> Set[Tuple[str, str, str, str]]:
    """Find similar author pairs from different platforms."""
    similar_pairs = set()
    platform_pair_counts = {}

    for i, j in zip(*np.where(average_similarity >= threshold)):
        if i < j:  # Only process each pair once
            author1, author2 = authors[i], authors[j]
            platform1 = author_platform_dict.get(author1, '')
            platform2 = author_platform_dict.get(author2, '')

            # Only add pairs from different platforms
            if platform1 and platform2 and platform1 != platform2:
                similar_pairs.add((author1, author2, platform1, platform2))

                # Count platform pairs
                pair = tuple(sorted([platform1, platform2]))
                platform_pair_counts[pair] = platform_pair_counts.get(pair, 0) + 1

    # Log platform pair distribution
    logging.info(f"\nPlatform pair distribution:")
    for (p1, p2), count in sorted(platform_pair_counts.items()):
        logging.info(f"{p1} <-> {p2}: {count} connections")

    return similar_pairs


def analyze_url_similarity_network() -> None:
    """Analyze URL sharing patterns across different platforms."""
    base_dir = Path("..")
    output_dir = base_dir / "networks" / "cross_platform"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NetworkConfig()
    logging.info(f"Starting cross-platform analysis with threshold {config.threshold}")

    # Load and combine data from all platforms
    all_data = []
    for platform in config.platforms:
        try:
            platform_df = load_platform_data(platform, base_dir)
            all_data.append(platform_df)
        except FileNotFoundError:
            logging.warning(f"No data found for {platform}")

    df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total records: {len(df)}")

    # Create author-platform mapping
    author_platforms = df[['author', 'platform']].drop_duplicates()
    author_platform_dict = dict(zip(author_platforms['author'], author_platforms['platform']))

    # Extract all author-URL pairs
    logging.info("Extracting URL pairs...")
    all_pairs = [(row['author'], url)
                 for _, row in tqdm(df.iterrows(), total=len(df))
                 for url in extract_urls(row['text'], config.url_pattern)]

    # Create global mappings
    authors = sorted({author for author, _ in all_pairs})
    urls = sorted({url for _, url in all_pairs})
    author_to_idx = {author: i for i, author in enumerate(authors)}
    url_to_idx = {url: i for i, url in enumerate(urls)}

    logging.info(f"Found {len(authors)} unique authors and {len(urls)} unique URLs")

    # Initialize accumulator for similarity matrices
    accumulated_similarity = np.zeros((len(authors), len(authors)))
    timeframe_count = 0

    # Process data by timeframes
    start_date = df['create_time'].min()
    end_date = df['create_time'].max()
    current_date = start_date

    logging.info(f"Processing data from {start_date} to {end_date}")
    with tqdm(total=(end_date - start_date).days) as pbar:
        while current_date <= end_date:
            timeframe_end = current_date + timedelta(days=config.timeframe_days)

            # Filter data for current timeframe
            mask = (df['create_time'] >= current_date) & (df['create_time'] < timeframe_end)
            timeframe_df = df[mask]

            if not timeframe_df.empty:
                timeframe_pairs = process_timeframe_data(timeframe_df, config)

                if timeframe_pairs:
                    # Calculate similarity matrix for current timeframe
                    freq_matrix = create_frequency_matrix(timeframe_pairs, author_to_idx, url_to_idx)
                    similarity = calculate_similarity_matrix(freq_matrix)

                    # Accumulate similarity matrix
                    accumulated_similarity += similarity.toarray()
                    timeframe_count += 1

            current_date = timeframe_end
            pbar.update(config.timeframe_days)

    logging.info(f"Processed {timeframe_count} timeframes with activity")

    # Calculate average similarity matrix
    average_similarity = accumulated_similarity / timeframe_count if timeframe_count > 0 else accumulated_similarity

    # Find cross-platform similar authors
    similar_pairs = find_cross_platform_pairs(
        average_similarity, authors, author_platform_dict, config.threshold
    )

    logging.info(f"Found {len(similar_pairs)} cross-platform similar author pairs")

    # Save results
    logging.info("Saving results...")

    # Add this before saving the similarity matrix in analyze_cross_platform_similarity():

    # Filter out edges between same-platform authors
    logging.info("Filtering out same-platform connections...")
    filtered_similarity = average_similarity.copy()

    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            author1, author2 = authors[i], authors[j]
            platform1 = author_platform_dict.get(author1, '')
            platform2 = author_platform_dict.get(author2, '')

            # Set similarity to 0 if authors are from the same platform
            if platform1 == platform2 or not platform1 or not platform2:
                filtered_similarity[i, j] = 0
                filtered_similarity[j, i] = 0  # Matrix is symmetric

    # Count filtered connections
    original_connections = np.count_nonzero(average_similarity)
    remaining_connections = np.count_nonzero(filtered_similarity)
    filtered_out = original_connections - remaining_connections

    logging.info(f"Original non-zero connections: {original_connections}")
    logging.info(f"Connections after filtering: {remaining_connections}")
    logging.info(f"Filtered out {filtered_out} same-platform connections")

    # Save the filtered similarity matrix
    filtered_similarity_matrix = csr_matrix(filtered_similarity)
    save_npz(str(output_dir / "author_similarity_matrix.npz"), filtered_similarity_matrix)

    # Save similar pairs with platform information
    similar_pairs_json = [
        {
            "author1": pair[0],
            "author2": pair[1],
            "platform1": pair[2],
            "platform2": pair[3]
        }
        for pair in similar_pairs
    ]

    with open(output_dir / "author_similar_authors.json", 'w') as f:
        json.dump(similar_pairs_json, f, indent=2)

    # Save mappings and metadata
    with open(output_dir / "author_mappings.json", 'w') as f:
        json.dump({
            'author_to_idx': author_to_idx,
            'url_to_idx': url_to_idx,
            'timeframe_count': timeframe_count,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'platforms': list(config.platforms),
            'author_platforms': author_platform_dict
        }, f, indent=2)

    logging.info("Cross-platform analysis completed successfully")


if __name__ == "__main__":
    analyze_url_similarity_network()