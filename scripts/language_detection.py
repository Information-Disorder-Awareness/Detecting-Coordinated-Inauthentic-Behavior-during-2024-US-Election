import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

from custom_types.Platform import Platform

tqdm.pandas()

classificator = pipeline("text-classification", model="ERCDiDip/langdetect", truncation=True)


def clean_text(text):
    text = re.sub(r'\\[ntr]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_lang(text):
    return classificator(text)[0]['label']


def detect_language(platform: Platform):
    df = pd.read_csv(f"./datasets/{platform}/filtered_posts.csv")

    print("Starting text cleaning...")
    df['text'] = df['text'].progress_apply(clean_text)

    print("Starting language identification...")

    df['lang'] = df['text'].progress_apply(detect_lang)

    df.to_csv(f"./datasets/{platform}/filtered_posts_cleaned.csv", index=False)
