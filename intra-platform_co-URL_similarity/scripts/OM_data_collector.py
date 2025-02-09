import requests
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz

from custom_types.Platform import Platform
from typing import TypeGuard
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

jwt_token = os.getenv('OM_KEY')
if not jwt_token:
    logging.error("OM_KEY environment variable not found")
    raise ValueError("OM_KEY environment variable not found")

author_keys = {
    '4chan': 'name',
    '8kun': 'name',
    'bluesky': 'author',
    'bitchute_comment': 'fullname',
    'fediverse': ['account', 'username'],
    'gab': ['account', 'username'],
    'gettr': 'uid',
    'kiwifarms': 'author_username',
    'lbry_comment': 'channel_name',
    'mewe': 'username',
    'minds': ['user', 'username'],
    'ok': 'author',
    'parler': 'username',
    'poal': 'sub',
    'rumble_comment': 'username',
    'rutube_comment': ['user', 'name'],
    'win': 'author',
    'telegram': 'channeltitle',
    'tiktok_comment': 'author',
    'truth_social': ['account', 'username'],
    'vk': 'author',
    'wimkin': 'author',
}


def is_platform(value: str) -> TypeGuard[Platform]:
    return value in Platform._value2member_map_


def fetch_interval_posts(start_time: datetime, end_time: datetime, site: str,
                         limit: int, term: str) -> dict:
    logging.info(f"Fetching posts for {site} from {start_time} to {end_time} with term: {term}")

    try:
        response = requests.get(
            'https://api.openmeasures.io/content',
            params={
                'site': site,
                'limit': limit,
                'term': term,
                'since': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'until': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            },
            headers={
                'Authorization': f'Bearer {jwt_token}',
            },
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching posts: {str(e)}")
        raise


def standardize_datetime(date_input):
    if isinstance(date_input, (int, str)) and str(date_input).isdigit():
        return datetime.fromtimestamp(int(date_input), tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')

    if isinstance(date_input, str):
        if '+00' in date_input and not date_input.endswith('00'):
            date_input = date_input + '00'

        if 'T' in date_input and '.' in date_input:
            parts = date_input.split('.')
            if len(parts) == 2:
                base, fraction = parts

                tz_split = None
                if '+' in fraction:
                    fraction, tz_split = fraction.split('+')
                elif '-' in fraction:
                    fraction, tz_split = fraction.split('-')

                fraction = fraction[:6]
                if tz_split:
                    tz_split = tz_split.ljust(4, '0')
                    date_input = f"{base}.{fraction}+{tz_split}"
                else:
                    date_input = f"{base}.{fraction}"

    formats = [
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%y(%a)%H:%M:%S'
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_input), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

    logging.error(f"Unable to parse date: {date_input}")
    raise ValueError(f"Unable to parse date: {date_input}")


def process_time_range(starting_df: pd.DataFrame, start_date: datetime, platform: Platform, terms: list,
                       end_date=datetime.strptime("2024-11-30", '%Y-%m-%d'), interval_hours=6, limit=10000) -> None:
    logging.info(f"Processing time range for {platform} from {start_date} to {end_date}")

    all_posts = starting_df
    current_time = start_date

    for term in terms:
        logging.info(f"Processing term: {term}")
        while current_time < end_date:
            try:
                next_time = current_time + timedelta(hours=interval_hours)
                if next_time > end_date:
                    next_time = end_date

                data = fetch_interval_posts(current_time, next_time, platform, limit, term)
                time_key = data.get('created_key', '')
                content_key = data.get('content_key', '')
                author_key = author_keys.get(platform, '')

                extracted_data = [
                    {
                        "create_time": standardize_datetime(item.get("_source", {}).get(time_key, np.NaN)),
                        "text": item.get("_source", {}).get(content_key, np.NaN),
                        "author": item.get("_source", {}).get(author_key[0], {}).get(author_key[1],
                                                                                     np.NaN) if platform in [
                            Platform.FEDIVERSE, Platform.GAB, Platform.MINDS, Platform.RUTUBE, Platform.TRUTH]
                        else item["_source"].get(author_key, np.NaN),
                        "label": term
                    }
                    for item in data.get("hits", {}).get("hits", {})
                ]

                # Adjust interval based on results
                if len(extracted_data) < 2000:
                    interval_hours *= 2
                    logging.info(f"Increasing interval to {interval_hours} hours due to low data volume")
                elif len(extracted_data) == 10000:
                    interval_hours /= 4
                    logging.warning(f"Query limit reached. Decreasing interval to {interval_hours} hours")
                    continue
                elif len(extracted_data) > 7000:
                    interval_hours /= 2
                    logging.info(f"Decreasing interval to {interval_hours} hours")
                elif len(extracted_data) > 4000:
                    interval_hours /= 1.5
                    logging.info(f"Slightly decreasing interval to {interval_hours} hours")

                # Update dataframe and save
                all_posts = pd.concat([all_posts, pd.DataFrame(extracted_data)],
                                      ignore_index=True).dropna().drop_duplicates(
                    subset=['create_time', 'text', 'author'])

                output_path = f"../datasets/{platform}/dataset.csv"
                all_posts.to_csv(output_path, index=False)

                logging.info(
                    f"Retrieved {len(extracted_data)} elements from {current_time} to {next_time} about {term}. Total records: {len(all_posts)}")
                current_time = next_time

            except Exception as e:
                logging.error(f"Error processing time range {current_time} to {next_time}: {str(e)}")
                raise

        interval_hours = 6
        current_time = datetime.strptime("2024-01-01", '%Y-%m-%d')


def gather_data(platform: Platform) -> None:
    logging.info("Starting data collection...")

    if not is_platform(str(platform)):
        logging.error(f"Invalid platform: {platform}")
        raise ValueError(f"Invalid platform: {platform}")

    start_date = datetime.strptime("2024-01-01", '%Y-%m-%d')
    terms = ["Trump", "trumpsupporters", "MAGA", "makeamericagreatagain", "JD Vance",
             "letsgobrandon", "WWG1WGA", "Harris", "Biden", "voteblue2024",
             "bidenharris2024", "US Elections", "2024 Elections", "2024 Presidential Elections"]
    df = pd.DataFrame()

    if not os.path.exists(f"../datasets/{platform}"):
        logging.info(f"Creating /datasets/{platform}")
        os.makedirs(f"../datasets/{platform}")
    else:
        logging.info(f"Directory datasets/{platform} already exists")
        if os.path.exists(f"../datasets/{platform}/dataset.csv"):
            df = pd.read_csv(f"../datasets/{platform}/dataset.csv")
            start_date = df['create_time'].iloc[-1]
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=1)
            last_term = df['label'].iloc[-1]
            terms = terms[terms.index(last_term):]

    logging.info(f"Start date: {start_date}")
    logging.info(f"Terms to process: {terms}")

    if not terms:
        logging.info("No terms to process, exiting")
        return

    process_time_range(df, start_date, platform, terms)
