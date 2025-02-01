from custom_types.Platform import Platform

from scripts.OM_data_collector import gather_data
from scripts.url_expander import unspool_texts
from scripts.urls_dataset_filter import process_urls
from scripts.urls_dataset_analysis import process_urls_stats
from scripts.filter_significant_urls import filter_relevant_urls
from scripts.filter_active_users import filter_active_users
from scripts.language_detection import detect_language
from scripts.similarity_network_generator import analyze_url_similarity_network


def main(platform: Platform) -> None:
    gather_data(platform)
    unspool_texts(platform)
    process_urls(platform)
    process_urls_stats(platform)
    filter_relevant_urls(platform)
    filter_active_users(platform)
    detect_language(platform)
    analyze_url_similarity_network(platform)


if __name__ == '__main__':
    main(Platform.TELEGRAM)
