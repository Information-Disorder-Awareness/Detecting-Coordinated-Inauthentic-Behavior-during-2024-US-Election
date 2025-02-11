import pandas as pd
import re
from urllib.parse import urlparse

from custom_types.Platform import Platform


def analyze_text_data(df):
    """
    Analyzes a DataFrame containing 'author' and 'text' columns to extract metrics.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'author' and 'text' columns

    Returns:
    dict: Dictionary containing the analysis metrics
    """
    # Validate input columns
    required_columns = {'author', 'text'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Basic metrics
    total_rows = len(df)
    unique_authors = df['author'].nunique()

    # URL extraction and analysis
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    # Extract all URLs from text
    all_urls = []
    for text in df['text']:
        if pd.isna(text):
            continue
        urls = re.findall(url_pattern, str(text))
        all_urls.extend(urls)

    total_urls = len(all_urls)

    # Extract and count unique domains
    domains = []
    for url in all_urls:
        try:
            domain = urlparse(url).netloc
            if domain:
                domains.append(domain)
        except Exception as e:
            print(f"Error parsing URL {url}: {e}")

    unique_domains = len(set(domains))

    # Prepare results
    results = {
        'total_rows': total_rows,
        'unique_authors': unique_authors,
        'total_urls': total_urls,
        'unique_domains': unique_domains
    }

    return results


# Example usage:
if __name__ == "__main__":
    for platform in [Platform.FEDIVERSE, Platform.GAB, Platform.MINDS, Platform.TELEGRAM, Platform.VK]:
        df = pd.read_csv(f"./datasets/{platform}/dataset.csv")
        results = analyze_text_data(df)

        print("="*50)
        print(f"Platform: {platform}")
        print(f"Total rows: {results['total_rows']}")
        print(f"Unique authors: {results['unique_authors']}")
        print(f"Total URLs: {results['total_urls']}")
        print(f"Unique domains: {results['unique_domains']}")