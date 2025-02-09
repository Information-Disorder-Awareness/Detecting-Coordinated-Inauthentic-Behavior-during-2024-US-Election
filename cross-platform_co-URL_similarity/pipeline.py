from scripts.similarity_network_generator import analyze_url_similarity_network
from scripts.threshold_calculator import calculate_thresholds
from scripts.network_visualizer import create_network_visualization
from scripts.coordinators_domain_analysis import get_coordinators_domains
from scripts.plot_domains import plot_domain_counts

def main() -> None:
    #analyze_url_similarity_network()
    #calculate_thresholds()
    create_network_visualization()
    #get_coordinators_domains()
    #plot_domain_counts()


if __name__ == '__main__':
    main()
