import os
import pandas as pd
from custom_types.Platform import Platform

def filter_urls(df: pd.DataFrame) -> pd.DataFrame:
    url_pattern = r'(?:https?://(?:www\.)?|(?<![\w])www\.)[\w\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'

    return df[df['text'].str.contains(url_pattern, regex=True, na=False)]

def process_urls(platform: Platform) -> None:
    df = pd.read_csv(f"./datasets/{platform}/dataset_unspooled.csv")

    if os.path.exists(f"./datasets/{platform}/dataset_urls.csv"):
        print("URLS already filtered")
        return

    print("Starting URLS filter...")
    filtered_df = filter_urls(df)

    filtered_df.to_csv(f"./datasets/{platform}/dataset_urls.csv", index=False)
    print(f"Correctly filtered {len(filtered_df)} urls in dataset/{platform}/dataset_urls.csv")
