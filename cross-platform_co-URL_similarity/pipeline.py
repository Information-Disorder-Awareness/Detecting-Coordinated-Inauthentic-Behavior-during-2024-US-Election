from scripts.similarity_network_generator import analyze_url_similarity_network
from scripts.threshold_calculator import calculate_thresholds
from scripts.network_visualizer import create_network_visualization


def main() -> None:
    analyze_url_similarity_network()
    calculate_thresholds()
    create_network_visualization()


if __name__ == '__main__':
    main()
