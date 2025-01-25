import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time
import warnings
import urllib3
from tqdm import tqdm
import re
from urllib.parse import urlparse, parse_qs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def is_shortened_url(url):
    url_patterns = [
        r'bit\.ly',
        r't\.co',
        r'tinyurl\.com',
        r'goo\.gl',
        r'ow\.ly',
        r'cutt\.ly',
        r'rebrand\.ly',
        r'tiny\.cc',
        r'is\.gd',
    ]
    return any(re.search(pattern, url) for pattern in url_patterns)


def normalize_url(url):
    """Normalize URL by removing certain query parameters and fragments"""
    parsed = urlparse(url)
    # Get base domain without www or subdomain
    domain = '.'.join(parsed.netloc.split('.')[-2:])
    path = parsed.path.rstrip('/')

    # Parse and filter query parameters
    query_params = parse_qs(parsed.query)
    filtered_params = {k: v for k, v in query_params.items()
                       if k.lower() not in ['utm_source', 'utm_medium', 'utm_campaign']}

    # Reconstruct URL
    normalized = f"{parsed.scheme}://{domain}{path}"
    if filtered_params:
        normalized += '?' + '&'.join(f"{k}={v[0]}" for k, v in sorted(filtered_params.items()))
    return normalized


def expand_url(url, session):
    if not is_shortened_url(url):
        return url
    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=10,
            verify=False
        )
        return normalize_url(response.url)
    except Exception as e:
        return url


def expand_urls_in_dataframe(df, url_column='url', max_workers=3, batch_size=50):
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in DataFrame")

    result_df = df.copy()
    # Normalize all URLs first
    result_df[f'expanded_{url_column}'] = df[url_column].fillna('').apply(normalize_url)

    valid_urls = df[url_column].dropna()
    valid_urls = valid_urls[valid_urls.str.startswith(('http://', 'https://'))]
    shortened_urls = valid_urls[valid_urls.apply(is_shortened_url)]
    total_urls = len(shortened_urls)

    if total_urls == 0:
        return result_df

    session = create_session()
    processed = 0

    for i in range(0, total_urls, batch_size):
        batch_urls = shortened_urls[i:i + batch_size]
        for url in tqdm(batch_urls,
                        desc=f"Processing URLs {processed}-{min(processed + batch_size, total_urls)} of {total_urls}"):
            expanded = expand_url(url, session)
            result_df.loc[
                result_df[url_column] == url,
                f'expanded_{url_column}'
            ] = expanded
            time.sleep(0.5)
            processed += 1

    return result_df

if __name__ == "__main__":
    df = pd.read_csv('../datasets/telegram_url_statistics.csv')[:100]

    expanded_df = expand_urls_in_dataframe(df)

    expanded_df.to_csv('expanded_urls.csv', index=False)
    print("\nResults saved to expanded_urls.csv")