import os
import pandas as pd
import re
from typing import Dict, Set
from dataclasses import dataclass
from urllib.parse import urlparse
from collections import defaultdict

from custom_types.Platform import Platform


@dataclass
class AuthorStats:
    total_unique_links: int
    links: list[str]

def normalize_url(url: str) -> str:
    if url.startswith('www.'):
        url = 'http://' + url

    parsed = urlparse(url)
    normalized_url = f"{parsed.scheme}://{parsed.netloc.replace('www.', '')}"

    if parsed.path and parsed.path != '/':
        normalized_url += parsed.path.rstrip('/')
    if parsed.query:
        normalized_url += f"?{parsed.query}"

    return normalized_url

def extract_urls(text: str) -> Set[str]:
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|www\.[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/[^\s]*)?'
    )

    urls = url_pattern.findall(str(text).lower())
    return {normalize_url(url) for url in urls}

def filter_active_linkers(df: pd.DataFrame, min_unique_links: int = 20) -> pd.DataFrame:
    author_links: Dict[str, Set[str]] = defaultdict(set)

    for author, group in df.groupby('author'):
        all_urls = set().union(*[extract_urls(text) for text in group['text']])
        if len(all_urls) >= min_unique_links:
            author_links[str(author)].update(all_urls)

    filtered_df = df[df['author'].isin(author_links)]

    return filtered_df

def filter_active_users(platform: Platform) -> None:

    if os.path.exists(f'../datasets/{platform}/filtered_posts.csv'):
        print("Dataset already filtered for active users")
        return

    df = pd.read_csv(f'../datasets/{platform}/dataset_relevant_urls.csv')

    print("Starting dataset filtering for active users...")
    filtered_df = filter_active_linkers(df)

    filtered_df.to_csv(f'../datasets/{platform}/filtered_posts.csv', index=False)
    print(f"Correctly filtered {len(filtered_df)} contents from active users in /datasets/{platform}/filtered_posts.csv")
