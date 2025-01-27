from custom_types.Platform import Platform

from scripts.OM_data_collector import gather_data
from scripts.urls_dataset_filter import process_urls
from scripts.urls_dataset_analysis import process_urls_stats
from scripts.filter_significant_urls import filter_relevant_urls
from scripts.language_detection import detect_language


def main(platform: Platform) -> None:
    gather_data(platform)
    process_urls(platform)
    process_urls_stats(platform)
    filter_relevant_urls(platform)
    detect_language(platform)


if __name__ == '__main__':
    main(Platform.TELEGRAM)
