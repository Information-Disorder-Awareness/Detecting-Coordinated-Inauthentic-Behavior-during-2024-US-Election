import os
import re
import pandas as pd
from custom_types.Platform import Platform


def get_relevant_urls(url_stats: pd.DataFrame) -> pd.Series:
    counts = url_stats['count']

    percentile_90 = counts.quantile(0.9)
    filtered = url_stats[(counts > percentile_90) & (counts > 5)]

    return filtered['url']


def filter_texts_with_urls(df: pd.DataFrame, relevant_urls: pd.Series) -> pd.DataFrame:
    escaped_urls = [re.escape(url) for url in relevant_urls]
    pattern = '|'.join(escaped_urls)
    return df[df['text'].str.contains(pattern, na=False, regex=True)]


def filter_relevant_urls(platform: Platform) -> None:
    df = pd.read_csv(f'./datasets/{platform}/url_stats.csv')
    relevant_urls = get_relevant_urls(df)

    if os.path.exists(f"./datasets/{platform}/dataset_relevant_urls.csv"):
        print("Dataset already filtered for relevant urls")
        return

    print("Starting dataset filtering for relevant urls...")
    df = pd.read_csv(f'./datasets/{platform}/dataset_urls.csv')
    filtered_df = filter_texts_with_urls(df, relevant_urls)

    filtered_df.to_csv(f'./datasets/{platform}/dataset_relevant_urls.csv', index=False)
    print(f"Correctly filtered {len(filtered_df)} contents with relevant URLs in dataset/{platform}/filtered_urls.csv")