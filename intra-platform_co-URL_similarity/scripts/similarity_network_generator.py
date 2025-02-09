from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import re
import json
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
    logging.debug(f"Creating frequency matrix for {len(pairs)} pairs")
    rows = [author_to_idx[author] for author, _ in pairs]
    cols = [url_to_idx[url] for _, url in pairs]
    matrix = csr_matrix(
        (np.ones(len(pairs)), (rows, cols)),
        shape=(len(author_to_idx), len(url_to_idx))
    )
    logging.debug(f"Created matrix with shape {matrix.shape}")
    return matrix


def calculate_similarity_matrix(freq_matrix):
    """Calculate similarity matrix from frequency matrix"""
    logging.debug("Calculating TF-IDF transformation")
    tfidf_matrix = TfidfTransformer(smooth_idf=True).fit_transform(freq_matrix)
    logging.debug("Normalizing matrix")
    normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    similarity = normalized @ normalized.T
    logging.debug(f"Calculated similarity matrix with shape {similarity.shape}")
    return similarity


def analyze_url_similarity_network(platform: str) -> None:
    logging.info(f"Starting URL similarity network analysis for platform: {platform}")

    # Setup paths
    base_dir = Path("..")
    input_file = base_dir / "datasets" / platform / "filtered_posts_cleaned.csv"
    output_dir = base_dir / "networks" / platform
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input file: {input_file}")
    logging.info(f"Output directory: {output_dir}")

    config = NetworkConfig()
    logging.info(f"Using configuration: threshold={config.threshold}, timeframe={config.timeframe_days} days")

    # Read and prepare data
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info("Reading input data")
    df = pd.read_csv(input_file)
    logging.info(f"Loaded {len(df)} posts")

    # Convert timestamp column
    df['create_time'] = pd.to_datetime(df['create_time'])
    date_range = f"{df['create_time'].min()} to {df['create_time'].max()}"
    logging.info(f"Data range: {date_range}")

    # Create global author and URL mappings
    logging.info("Extracting author-URL pairs")
    all_pairs = [(row['author'], url)
                 for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting pairs")
                 for url in extract_urls(row['text'], config.url_pattern)]

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

    logging.info("Processing data in timeframes")
    with tqdm(total=(end_date - start_date).days, desc="Processing timeframes") as pbar:
        while current_date <= end_date:
            timeframe_end = current_date + timedelta(days=config.timeframe_days)

            # Filter data for current timeframe
            mask = (df['create_time'] >= current_date) & (df['create_time'] < timeframe_end)
            timeframe_df = df[mask]

            if not timeframe_df.empty:
                logging.debug(f"Processing timeframe {current_date} to {timeframe_end}")
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
                    logging.debug(f"Processed timeframe {timeframe_count} with {len(timeframe_pairs)} pairs")

            current_date = timeframe_end
            pbar.update(config.timeframe_days)

    logging.info(f"Processed {timeframe_count} timeframes with activity")

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

    logging.info(f"Found {len(similar_pairs)} similar author pairs")

    # Save results
    logging.info("Saving results")

    # Save similarity matrix
    matrix_path = output_dir / "author_similarity_matrix.npz"
    average_similarity_matrix = csr_matrix(average_similarity)
    save_npz(str(matrix_path), average_similarity_matrix)
    logging.info(f"Saved similarity matrix to {matrix_path}")

    # Save similar author groups
    groups_path = output_dir / "similar_author_groups.json"
    with open(groups_path, 'w') as f:
        json.dump([list(pair) for pair in similar_pairs], f, indent=2)
    logging.info(f"Saved author groups to {groups_path}")

    # Save mappings
    mappings_path = output_dir / "author_url_mappings.json"
    with open(mappings_path, 'w') as f:
        json.dump({
            'author_to_idx': author_to_idx,
            'url_to_idx': url_to_idx,
            'timeframe_count': timeframe_count,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }, f, indent=2)
    logging.info(f"Saved mappings to {mappings_path}")

    logging.info("Analysis completed successfully")
