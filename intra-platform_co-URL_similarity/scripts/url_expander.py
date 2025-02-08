import os.path
from tqdm import tqdm
import pandas as pd
from unspooler import *
from typing import Dict
from custom_types.Platform import Platform

tqdm.pandas()


def replace_substrings(df, replacement_dict: Dict[str, str]):
    def replace_all(text):
        if pd.isna(text):
            return text

        text = str(text)

        result = text
        for key in sorted(replacement_dict.keys(), key=len, reverse=True):
            pattern = str(key)
            replacement = str(replacement_dict[key])
            result = result.replace(pattern, replacement)
        return result

    df['text'] = df['text'].progress_apply(replace_all)

    return df

def unspool_texts(platform: Platform):
    df = pd.read_csv(f'../datasets/{platform}/dataset.csv')
    if os.path.exists(f"../datasets/{platform}/dataset_unspooled.csv"):
        print("Dataset already unspooled.")
        return

    print("Unspooling shortened urls...")
    unspooled = unspool_easy(df['text'])

    print("Replacing shortened urls in the dataset...")
    df = replace_substrings(df, unspooled['urls'])

    df.to_csv(f'../datasets/{platform}/dataset_unspooled.csv', index=False)
    print(f"Correctly saved the unspooled dataset in /datasets/{platform}/dataset_unspooled.csv.")
