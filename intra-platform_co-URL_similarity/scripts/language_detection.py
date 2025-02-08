import os.path
import re
import pandas as pd
from tqdm import tqdm

from custom_types.Platform import Platform

tqdm.pandas()

def clean_text(text):
    text = re.sub(r'\\[ntr]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataset(platform: Platform):
    df = pd.read_csv(f"../datasets/{platform}/filtered_posts.csv")

    if os.path.exists(f"../datasets/{platform}/filtered_posts_cleaned.csv"):
        print("Dataset already cleaned.")
        return

    print("Starting text cleaning...")
    df['text'] = df['text'].progress_apply(clean_text)

    df.to_csv(f"../datasets/{platform}/filtered_posts_cleaned.csv", index=False)
