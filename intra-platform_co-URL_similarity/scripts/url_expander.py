import os.path
from tqdm import tqdm
import pandas as pd
import logging
from unspooler import *
from typing import Dict
from custom_types.Platform import Platform

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

tqdm.pandas()


def replace_substrings(df, replacement_dict: Dict[str, str]):
    """Replace shortened URLs with their expanded versions in the text."""
    logging.info(f"Starting URL replacement for {len(replacement_dict)} URLs")

    def replace_all(text):
        if pd.isna(text):
            return text

        text = str(text)
        result = text
        for key in sorted(replacement_dict.keys(), key=len, reverse=True):
            pattern = str(key)
            replacement = str(replacement_dict[key])
            result = result.replace(pattern, replacement)
        return result

    logging.info("Replacing URLs in text content")
    df['text'] = df['text'].progress_apply(replace_all)

    # Calculate replacement statistics
    total_replacements = sum(
        str(text).count(key)
        for text in df['text']
        for key in replacement_dict.keys()
    )
    logging.info(f"Completed {total_replacements} URL replacements")

    return df


def unspool_texts(platform: Platform):
    """Expand shortened URLs in the dataset."""
    logging.info(f"Starting URL expansion for platform: {platform}")

    input_path = f'../datasets/{platform}/dataset.csv'
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = f"../datasets/{platform}/dataset_unspooled.csv"
    if os.path.exists(output_path):
        logging.info("Dataset already unspooled")
        return

    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")

    logging.info("Unspooling shortened URLs")
    unspooled = unspool_easy(df['text'])
    logging.info(f"Found {len(unspooled['urls'])} unique shortened URLs to expand")

    df = replace_substrings(df, unspooled['urls'])

    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved unspooled dataset to {output_path}")
    logging.info("URL expansion completed")
