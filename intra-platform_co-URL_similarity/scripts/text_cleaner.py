import os.path
import re
import pandas as pd
import logging
from tqdm import tqdm

from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

tqdm.pandas()


def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    # Remove escape sequences
    text = re.sub(r'\\[ntr]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataset(platform: Platform):
    logging.info(f"Starting dataset cleaning for platform: {platform}")

    input_path = f"../datasets/{platform}/filtered_posts.csv"
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = f"../datasets/{platform}/filtered_posts_cleaned.csv"
    if os.path.exists(output_path):
        logging.info("Dataset already cleaned")
        return

    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    logging.info("Cleaning text content")
    df['text'] = df['text'].progress_apply(clean_text)

    # Calculate cleaning statistics
    non_empty_posts = df['text'].str.strip().str.len() > 0
    posts_with_content = non_empty_posts.sum()
    logging.info(f"Posts with content after cleaning: {posts_with_content} ({posts_with_content / len(df) * 100:.2f}%)")

    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved cleaned dataset to {output_path}")
    logging.info("Text cleaning completed")
