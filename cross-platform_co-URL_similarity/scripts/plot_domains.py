import pandas as pd
import matplotlib.pyplot as plt

from custom_types.Platform import Platform


def plot_domain_counts():
    # Read and prepare data
    df = pd.read_csv("../datasets/cross_platform/domain_occurrences.csv", names=['domain', 'count'])[1:]
    df['count'] = pd.to_numeric(df['count'])

    top_20 = df.nlargest(20, 'count')

    # Create simple bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_20)), top_20['count'])

    # Set x-ticks with domain names
    plt.xticks(range(len(top_20)), top_20['domain'], rotation=45, ha='right')

    # Labels and title
    plt.xlabel('Domains')
    plt.ylabel('Count')
    plt.title('Top 20 Domain Occurrences')

    # Adjust layout
    plt.tight_layout()

    # Save and show
    plt.savefig("../networks/cross_platform/output/domain_plot.png", dpi=300, bbox_inches='tight')