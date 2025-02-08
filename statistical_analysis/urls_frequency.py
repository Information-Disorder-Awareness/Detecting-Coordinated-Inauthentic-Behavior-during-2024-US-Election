import requests
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from custom_types.Platform import Platform

load_dotenv()
jwt_token = os.getenv('OM_KEY')

def get_authors(platform: Platform) -> pd.DataFrame:
    response = requests.get(
        f'https://api.openmeasures.io/content',
        params={
            'site': platform,
            'limit': 10000,
            'term': 'Trump',
            'since': datetime.strptime("2024-01-01", '%Y-%m-%d'),
            'until': datetime.strptime("2024-11-30", '%Y-%m-%d'),
        },
        headers={
            'Authorization': f'Bearer {jwt_token}',
        },
    )
    response.raise_for_status()
    data = response.json()

    content_key = data.get('content_key', '')
    extracted_data = [
        {
            "text": item.get("_source", {}).get(content_key, np.NaN),
        }
        for item in data.get("hits", {}).get("hits", {})
    ]

    return pd.DataFrame(extracted_data)

def filter_urls(df: pd.DataFrame) -> pd.DataFrame:
    url_pattern = r'(?:https?://(?:www\.)?|(?<![\w])www\.)[\w\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'

    return df[df['text'].str.contains(url_pattern, regex=True, na=False)]


if __name__ == '__main__':
    for platfrom in Platform:
        df = get_authors(platfrom)

        df_filtered = filter_urls(df)

        print(f"Percentuale di url presenti nei contenuti di {platfrom}: {len(df_filtered)/len(df)*100}%")