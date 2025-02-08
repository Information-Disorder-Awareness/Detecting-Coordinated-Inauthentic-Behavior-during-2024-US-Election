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
from datetime import datetime, timedelta


@dataclass(frozen=True)
class NetworkConfig:
    threshold: float = 0.5
    batch_size: int = 10000
    timeframe_days: int = 7
    url_pattern: str = r'https?://[^\s<>"]+|www\.[^\s<>"]+|bit\.ly/[^\s<>"]+|t\.co/[^\s<>"]+|tinyurl\.com/[^\s<>"]+|is\.gd/[^\s<>"]+'


def extract_urls(text: str, pattern: str) -> set:
    if not isinstance(text, str):
        return set()
    urls = re.findall(pattern, text, re.IGNORECASE)
    return {url.lower().rstrip('/.').replace('www.', '') for url in urls}


def create_frequency_matrix(pairs, author_to_idx, url_to_idx):
    """Create a frequency matrix from author-URL pairs"""
    rows = [author_to_idx[author] for author, _ in pairs]
    cols = [url_to_idx[url] for _, url in pairs]
    return csr_matrix(
        (np.ones(len(pairs)), (rows, cols)),
        shape=(len(author_to_idx), len(url_to_idx))
    )


def calculate_similarity_matrix(freq_matrix):
    """Calculate similarity matrix from frequency matrix"""
    tfidf_matrix = TfidfTransformer(smooth_idf=True).fit_transform(freq_matrix)
    normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    return normalized @ normalized.T


def analyze_url_similarity_network(platform: str) -> None:
    # Setup paths
    base_dir = Path("..")
    input_file = base_dir / "datasets" / platform / "filtered_posts_cleaned.csv"
    output_dir = base_dir / "networks" / platform
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NetworkConfig()

    # Read and prepare data
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    # Verifica delle colonne presenti
    print(f"Columns in dataset: {df.columns.tolist()}")

    # Conversione della colonna temporale
    df['create_time'] = pd.to_datetime(df['create_time'])
    print(f"Data range: from {df['create_time'].min()} to {df['create_time'].max()}")

    # Create global author and URL mappings
    print("Extracting URL pairs...")
    all_pairs = [(row['author'], url)
                 for _, row in tqdm(df.iterrows(), total=len(df))
                 for url in extract_urls(row['text'], config.url_pattern)]

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

    # Find similar authors using average similarity
    similar_pairs = {
        tuple(sorted([authors[i], authors[j]]))
        for i, j in zip(*np.where(average_similarity >= config.threshold))
        if i < j
    }

    print(f"Found {len(similar_pairs)} similar author pairs")

    # Save results
    print("Saving results...")
    average_similarity_matrix = csr_matrix(average_similarity)
    save_npz(str(output_dir / "author_similarity_matrix.npz"), average_similarity_matrix)

    with open(output_dir / "similar_author_groups.json", 'w') as f:
        json.dump([list(pair) for pair in similar_pairs], f, indent=2)

    with open(output_dir / "author_url_mappings.json", 'w') as f:
        json.dump({
            'author_to_idx': author_to_idx,
            'url_to_idx': url_to_idx,
            'timeframe_count': timeframe_count,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }, f, indent=2)

    print("Analysis completed successfully")


if __name__ == "__main__":
    analyze_url_similarity_network("TELEGRAM")