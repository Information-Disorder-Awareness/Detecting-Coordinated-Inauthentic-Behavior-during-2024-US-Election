import os
import pandas as pd
import logging
from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def filter_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to keep only posts containing URLs."""
    url_pattern = r'(?:https?://(?:www\.)?|(?<![\w])www\.)[\w\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'

    filtered_df = df[df['text'].str.contains(url_pattern, regex=True, na=False)]

    # Calculate filtering statistics
    total_posts = len(df)
    posts_with_urls = len(filtered_df)
    retention_rate = (posts_with_urls / total_posts) * 100

    logging.info(f"Found {posts_with_urls} posts containing URLs")
    logging.info(f"Retention rate: {retention_rate:.2f}% of original posts")

    return filtered_df


def process_urls(platform: Platform) -> None:
    """Process dataset to filter posts containing URLs."""
    logging.info(f"Starting URL filtering for platform: {platform}")

    input_path = f"../datasets/{platform}/dataset_unspooled.csv"
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = f"../datasets/{platform}/dataset_urls.csv"
    if os.path.exists(output_path):
        logging.info("URLs already filtered")
        return

    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    filtered_df = filter_urls(df)

    # Calculate author statistics
    original_authors = df['author'].nunique()
    filtered_authors = filtered_df['author'].nunique()
    author_retention = (filtered_authors / original_authors) * 100
    logging.info(f"Authors with URLs: {filtered_authors} ({author_retention:.2f}% of original authors)")

    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved filtered dataset to {output_path}")
    logging.info("URL filtering completed")

