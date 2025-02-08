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

    extracted_data = [
        {
            "author": item.get("_source", {}).get('name', np.NaN),
        }
        for item in data.get("hits", {}).get("hits", {})
    ]

    return pd.DataFrame(extracted_data)


if __name__ == '__main__':
    for platform in [Platform.FOURCHAN, Platform.EIGHTKUN]:
        authors = get_authors(platform)

        counts = authors['author'].value_counts().head(3)

        print(f"Authors for {platform}:")
        for author, count in counts.items():
            print(f"{author}: {count} ({count/len(authors) * 100}%)")
