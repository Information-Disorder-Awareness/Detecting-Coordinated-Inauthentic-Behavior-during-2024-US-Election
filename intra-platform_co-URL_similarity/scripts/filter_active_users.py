import os
import pandas as pd
import re
import logging
from typing import Dict, Set
from dataclasses import dataclass
from urllib.parse import urlparse
from collections import defaultdict
from tqdm import tqdm

from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class AuthorStats:
    total_unique_links: int
    links: list[str]


def normalize_url(url: str) -> str:
    try:
        if url.startswith('www.'):
            url = 'http://' + url

        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc.replace('www.', '')}"

        if parsed.path and parsed.path != '/':
            normalized_url += parsed.path.rstrip('/')
        if parsed.query:
            normalized_url += f"?{parsed.query}"

        return normalized_url
    except ValueError:
        logging.warning(f"Invalid URL encountered: {url}")
        return ''  # Return empty string for invalid URLs


def extract_urls(text: str) -> Set[str]:
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|www\.[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/[^\s]*)?'
    )

    urls = url_pattern.findall(str(text).lower())
    normalized_urls = {url for url in (normalize_url(url) for url in urls) if url}
    return normalized_urls


def filter_active_linkers(df: pd.DataFrame, min_unique_links: int = 20) -> pd.DataFrame:
    logging.info(f"Filtering users with minimum {min_unique_links} unique links")
    author_links: Dict[str, Set[str]] = defaultdict(set)

    total_authors = df['author'].nunique()
    logging.info(f"Processing {total_authors} unique authors")

    for author, group in tqdm(df.groupby('author'), desc="Processing authors"):
        all_urls = set().union(*[extract_urls(text) for text in group['text']])
        if len(all_urls) >= min_unique_links:
            author_links[str(author)].update(all_urls)

    active_authors = len(author_links)
    logging.info(f"Found {active_authors} active authors ({active_authors / total_authors * 100:.2f}% of total)")

    filtered_df = df[df['author'].isin(author_links)]
    logging.info(
        f"Retained {len(filtered_df)} posts from active authors ({len(filtered_df) / len(df) * 100:.2f}% of total posts)")

    return filtered_df


def filter_active_users(platform: Platform) -> None:
    logging.info(f"Starting active user filtering for platform: {platform}")

    if os.path.exists(f'../datasets/{platform}/filtered_posts.csv'):
        logging.info("Dataset already filtered for active users")
        return

    input_path = f'../datasets/{platform}/dataset_relevant_urls.csv'
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    filtered_df = filter_active_linkers(df)

    output_path = f'../datasets/{platform}/filtered_posts.csv'
    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved filtered dataset to {output_path}")
    logging.info("Active user filtering completed")
