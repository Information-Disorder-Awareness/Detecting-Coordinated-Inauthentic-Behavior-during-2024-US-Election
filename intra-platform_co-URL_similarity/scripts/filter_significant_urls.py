import os
import re
import pandas as pd
import logging
from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_relevant_urls(url_stats: pd.DataFrame) -> pd.Series:
    counts = url_stats['count']

    percentile_90 = counts.quantile(0.9)
    logging.info(f"URL count 90th percentile threshold: {percentile_90}")

    filtered = url_stats[(counts > percentile_90) & (counts > 5)]
    logging.info(f"Found {len(filtered)} URLs above threshold (out of {len(url_stats)} total)")

    return filtered['url']


def filter_texts_with_urls(df: pd.DataFrame, relevant_urls: pd.Series) -> pd.DataFrame:
    logging.info(f"Creating regex pattern from {len(relevant_urls)} URLs")
    escaped_urls = [re.escape(url) for url in relevant_urls]
    pattern = '|'.join(escaped_urls)

    filtered_df = df[df['text'].str.contains(pattern, na=False, regex=True)]
    retention_rate = len(filtered_df) / len(df) * 100
    logging.info(f"Retained {len(filtered_df)} posts out of {len(df)} ({retention_rate:.2f}%)")

    return filtered_df


def filter_relevant_urls(platform: Platform) -> None:
    logging.info(f"Starting URL filtering for platform: {platform}")

    stats_path = f'../datasets/{platform}/url_stats.csv'
    if not os.path.exists(stats_path):
        logging.error(f"URL stats file not found: {stats_path}")
        raise FileNotFoundError(f"URL stats file not found: {stats_path}")

    logging.info(f"Loading URL statistics from {stats_path}")
    df = pd.read_csv(stats_path)
    logging.info(f"Loaded statistics for {len(df)} unique URLs")

    relevant_urls = get_relevant_urls(df)

    output_path = f"../datasets/{platform}/dataset_relevant_urls.csv"
    if os.path.exists(output_path):
        logging.info("Dataset already filtered for relevant URLs")
        return

    input_path = f'../datasets/{platform}/dataset_urls.csv'
    if not os.path.exists(input_path):
        logging.error(f"Dataset file not found: {input_path}")
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    logging.info(f"Loading main dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    filtered_df = filter_texts_with_urls(df, relevant_urls)

    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved filtered dataset to {output_path}")
    logging.info("URL filtering completed")
