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
from datetime import  timedelta
from typing import List

from scripts.threshold_calculator import find_maximum_increment_coordinates
from custom_types.Platform import Platform


@dataclass(frozen=True)
class NetworkConfig:
    threshold: float = 0.5
    batch_size: int = 10000
    timeframe_days: int = 7
    url_pattern: str = r'https?://[^\s<>"]+|www\.[^\s<>"]+|bit\.ly/[^\s<>"]+|t\.co/[^\s<>"]+|tinyurl\.com/[^\s<>"]+|is\.gd/[^\s<>"]+'
    platforms: List[str] = (Platform.TELEGRAM, Platform.GAB, Platform.GETTR, Platform.TELEGRAM, Platform.VK)  # Add your platforms here


def extract_urls(text: str, pattern: str) -> set:
    """Extract and normalize URLs from text."""
    if not isinstance(text, str):
        return set()
    urls = re.findall(pattern, text, re.IGNORECASE)
    return {url.lower().rstrip('/.').replace('www.', '') for url in urls}


def create_frequency_matrix(pairs, author_to_idx, url_to_idx):
    """Create a sparse frequency matrix from author-URL pairs."""
    rows = [author_to_idx[author] for author, _ in pairs]
    cols = [url_to_idx[url] for _, url in pairs]
    return csr_matrix(
        (np.ones(len(pairs)), (rows, cols)),
        shape=(len(author_to_idx), len(url_to_idx))
    )


def calculate_similarity_matrix(freq_matrix):
    """Calculate similarity matrix using TF-IDF and cosine similarity."""
    tfidf_matrix = TfidfTransformer(smooth_idf=True).fit_transform(freq_matrix)
    normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    return normalized @ normalized.T


def load_platform_data(platform: str, base_dir: Path) -> pd.DataFrame:
    """Load and prepare data for a specific platform."""
    input_file = base_dir / "datasets" / platform / "filtered_posts_cleaned.csv"
    df = pd.read_csv(input_file)
    df['platform'] = platform  # Add platform identifier
    df['create_time'] = pd.to_datetime(df['create_time'])
    return df


def analyze_cross_platform_similarity() -> None:
    """Analyze URL sharing patterns across different platforms."""
    # Setup paths
    base_dir = Path("..")
    output_dir = base_dir / "networks" / "cross_platform"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NetworkConfig()

    # Load and combine data from all platforms
    print("Loading data from all platforms...")
    all_data = []
    for platform in config.platforms:
        try:
            platform_df = load_platform_data(platform, base_dir)
            all_data.append(platform_df)
            print(f"Loaded {len(platform_df)} records from {platform}")
        except FileNotFoundError:
            print(f"Warning: No data found for {platform}")

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(df)}")

    # Create author-platform mapping for later filtering
    author_platforms = df[['author', 'platform']].drop_duplicates()
    author_platform_dict = dict(zip(author_platforms['author'], author_platforms['platform']))

    # Extract all author-URL pairs
    print("Extracting URL pairs...")
    all_pairs = [(row['author'], url)
                 for _, row in tqdm(df.iterrows(), total=len(df))
                 for url in extract_urls(row['text'], config.url_pattern)]

    # Create global mappings
    authors = sorted({author for author, _ in all_pairs})
    urls = sorted({url for _, url in all_pairs})
    author_to_idx = {author: i for i, author in enumerate(authors)}
    url_to_idx = {url: i for i, url in enumerate(urls)}

    print(f"Found {len(authors)} unique authors and {len(urls)} unique URLs")

    # Initialize accumulator for similarity matrices
    accumulated_similarity = np.zeros((len(authors), len(authors)))
    timeframe_count = 0

    # Process data by timeframes
    start_date = df['create_time'].min()
    end_date = df['create_time'].max()
    current_date = start_date

    print(f"Processing data from {start_date} to {end_date}")
    with tqdm(total=(end_date - start_date).days) as pbar:
        while current_date <= end_date:
            timeframe_end = current_date + timedelta(days=config.timeframe_days)

            # Filter data for current timeframe
            mask = (df['create_time'] >= current_date) & (df['create_time'] < timeframe_end)
            timeframe_df = df[mask]

            if not timeframe_df.empty:
                # Extract pairs for current timeframe
                timeframe_pairs = [(row['author'], url)
                                   for _, row in timeframe_df.iterrows()
                                   for url in extract_urls(row['text'], config.url_pattern)]

                if timeframe_pairs:
                    # Calculate similarity matrix for current timeframe
                    freq_matrix = create_frequency_matrix(timeframe_pairs, author_to_idx, url_to_idx)
                    similarity = calculate_similarity_matrix(freq_matrix)

                    # Accumulate similarity matrix
                    accumulated_similarity += similarity.toarray()
                    timeframe_count += 1

            current_date = timeframe_end
            pbar.update(config.timeframe_days)

    print(f"Processed {timeframe_count} timeframes with activity")

    # Calculate average similarity matrix
    if timeframe_count > 0:
        average_similarity = accumulated_similarity / timeframe_count
    else:
        average_similarity = accumulated_similarity

    # Find cross-platform similar authors
    similar_pairs = {
        (authors[i], authors[j], author_platform_dict[authors[i]], author_platform_dict[authors[j]])
        for i, j in zip(*np.where(average_similarity >= config.threshold))
        if i < j
    }

    print(f"Found {len(similar_pairs)} cross-platform similar author pairs")

    # Save results
    print("Saving results...")
    average_similarity_matrix = csr_matrix(average_similarity)
    save_npz(str(output_dir / "author_similarity_matrix.npz"), average_similarity_matrix)

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
            'platforms': list(config.platforms)
        }, f, indent=2)

    #
    print("Cross-platform analysis completed successfully")


if __name__ == "__main__":
    #analyze_cross_platform_similarity()
    find_maximum_increment_coordinates('cross_platform')