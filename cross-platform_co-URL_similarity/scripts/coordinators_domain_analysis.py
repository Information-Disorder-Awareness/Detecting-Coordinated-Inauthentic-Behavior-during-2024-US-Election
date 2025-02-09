import json
import os

import pandas as pd
import re

from custom_types.Platform import Platform

def load_dataframes() -> pd.DataFrame:
    total_df = pd.DataFrame()
    for platform in [Platform.FEDIVERSE, Platform.GAB, Platform.TELEGRAM, Platform.VK, Platform.MINDS]:
        platform_df = pd.read_csv(f"../datasets/{platform}/filtered_posts_cleaned.csv")
        total_df = pd.concat([total_df, platform_df])

    return total_df


def load_authors(file_path):
    """
    Load author names from the JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract author names from the authors dictionary
    author_names = [author_data['name'] for author_data in data['authors'].values()]
    return author_names


def extract_domains(text):
    """
    Extract domains from text using regex pattern
    Returns a list of domains
    """
    # Pattern to match URLs
    url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

    # Find all matches in text
    domains = re.findall(url_pattern, text)
    return domains if domains else []


def process_domains(df, author_names):
    """
    Process domains for specified authors and count occurrences
    """
    # Filter dataframe for specified authors
    filtered_df = df[df['author'].isin(author_names)]

    # Extract domains from each text entry
    all_domains = []
    for text in filtered_df['text']:
        if isinstance(text, str):  # Check if text is a string
            domains = extract_domains(text)
            all_domains.extend(domains)

    # Count domain occurrences
    domain_counts = pd.Series(all_domains).value_counts()
    return domain_counts


def get_coordinators_domains():
    # Load author names
    author_names = load_authors(f"../networks/cross_platform/output/network_authors.json")

    # Load your dataframe (you'll need to replace this with your actual dataframe loading)
    # Assuming your dataframe is stored in a CSV file
    df = load_dataframes()

    # Process domains and get counts
    domain_counts = process_domains(df, author_names)

    # Save results to file
    os.makedirs('../datasets/cross_platform', exist_ok=True)
    domain_counts.to_csv(f"../datasets/cross_platform/domain_occurrences.csv")

    print(f"Domain analysis complete. Found {len(domain_counts)} unique domains.")
    print(f"Results saved to 'domain_occurrences.csv'")
