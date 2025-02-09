import os
import pandas as pd
import re
import logging
from collections import Counter
from typing import Tuple
from tqdm import tqdm
from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_urls(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Analyze URLs in the dataset and return statistics."""
    pattern = r'(?:https?://(?:www\.)?|(?<![\w])www\.)[\w\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'

    logging.info("Extracting URLs from text content")
    all_urls = []
    for text in tqdm(df['text'], desc="Processing texts"):
        urls = re.findall(pattern, str(text))
        all_urls.extend(urls)

    total_urls = len(all_urls)
    logging.info(f"Found {total_urls} total URL occurrences")

    url_counts = Counter(all_urls)
    unique_urls = len(url_counts)
    logging.info(f"Found {unique_urls} unique URLs")

    # Calculate frequency statistics
    most_common = url_counts.most_common(5)
    logging.info("Top 5 most frequent URLs:")
    for url, count in most_common:
        logging.info(f"  {url}: {count} occurrences")

    url_stats = pd.DataFrame.from_dict(url_counts, orient='index', columns=['count'])
    url_stats.index.name = 'url'
    url_stats = url_stats.sort_values('count', ascending=False)

    # Calculate distribution statistics
    url_stats['percentage'] = (url_stats['count'] / total_urls) * 100
    single_occurrence = (url_stats['count'] == 1).sum()
    logging.info(f"URLs appearing only once: {single_occurrence} ({single_occurrence / unique_urls * 100:.2f}%)")

    return url_stats, unique_urls


def process_urls_stats(platform: Platform) -> None:
    """Process and save URL statistics for the given platform."""
    logging.info(f"Starting URL analysis for platform: {platform}")

    input_path = f"../datasets/{platform}/dataset_urls.csv"
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = f"../datasets/{platform}/url_stats.csv"
    if os.path.exists(output_path):
        logging.info("URL statistics already calculated")
        return

    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    url_stats, unique_count = analyze_urls(df)

    # Save results
    url_stats.to_csv(output_path)
    logging.info(f"Successfully saved URL statistics to {output_path}")

    # Log summary statistics
    posts_with_urls = df['text'].str.contains('http|www\.', na=False).sum()
    logging.info(f"Summary statistics:")
    logging.info(f"  Total posts: {len(df)}")
    logging.info(f"  Posts containing URLs: {posts_with_urls} ({posts_with_urls / len(df) * 100:.2f}%)")
    logging.info(f"  Unique URLs found: {unique_count}")
    logging.info("URL analysis completed")
