import os
import pandas as pd
import re
from collections import Counter
from typing import Tuple
from custom_types.Platform import Platform


def analyze_urls(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    pattern = r'(?:https?://(?:www\.)?|(?<![\w])www\.)[\w\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'

    all_urls = []
    for text in df['text']:
        urls = re.findall(pattern, str(text))
        all_urls.extend(urls)

    url_counts = Counter(all_urls)
    url_stats = pd.DataFrame.from_dict(url_counts, orient='index', columns=['count'])
    url_stats.index.name = 'url'
    url_stats = url_stats.sort_values('count', ascending=False)

    return url_stats, len(url_stats)


def process_urls_stats(platform: Platform) -> None:
    df = pd.read_csv(f"./datasets/{platform}/dataset_urls.csv")

    if os.path.exists(f"./datasets/{platform}/url_stats.csv"):
        print("Unique URLs already found")
        return

    print("Starting unique URLs research...")
    url_stats, unique_count = analyze_urls(df)

    url_stats.to_csv(f"./datasets/{platform}/url_stats.csv")
    print(f"Found {unique_count} unique URLs in datasets/{platform}/url_stats.csv")
