import json
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from custom_types.Platform import Platform


def load_dataframes() -> pd.DataFrame:
    total_df = pd.DataFrame()
    for platform in [Platform.FEDIVERSE, Platform.GAB, Platform.TELEGRAM, Platform.VK, Platform.MINDS]:
        platform_df = pd.read_csv(f"./datasets/{platform}/filtered_posts_cleaned.csv")
        total_df = pd.concat([total_df, platform_df])
    return total_df


def load_authors(file_path):
    """Load author names from the JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    author_names = [author_data['name'] for author_data in data['authors'].values()]
    return author_names


def extract_domains(text):
    """Extract domains from text using regex pattern"""
    url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    domains = re.findall(url_pattern, text)
    return domains if domains else []


def process_domains(df, filter_authors=None):
    """
    Process domains and count occurrences
    If filter_authors is provided, only count domains from those authors
    """
    if filter_authors is not None:
        df = df[df['author'].isin(filter_authors)]

    all_domains = []
    for text in df['text']:
        if isinstance(text, str):
            domains = extract_domains(text)
            all_domains.extend(domains)

    domain_counts = pd.Series(all_domains).value_counts()
    return domain_counts


def create_comparison_plot(platform, comparison_df):
    """
    Create and save the comparison visualization for top 20 domains by volume
    """
    # Get top 20 domains by total volume
    top_20 = comparison_df.nlargest(20, 'pre_filter')

    plt.figure(figsize=(15, 8))

    # Bar positions
    x = range(len(top_20))
    width = 0.35

    # Create bars
    plt.bar(x, top_20['pre_filter'], width,
            label='Pre-filtering', color='skyblue')
    plt.bar([i + width for i in x], top_20['post_filter'], width,
            label='Post-filtering', color='lightcoral')

    # Add percentage labels
    for i, row in enumerate(top_20.itertuples()):
        if row.pre_filter > 0:
            percentage = (row.post_filter / row.pre_filter) * 100
            plt.text(i + width, row.post_filter,
                     f'{percentage:.1f}%',
                     ha='center', va='bottom')

    # Customize plot
    plt.xlabel('Domains')
    plt.ylabel('Number of occurrences')
    plt.title('Top 20 Domains Comparison: Pre vs Post Filtering')
    plt.xticks([i + width / 2 for i in x],
               top_20.index,
               rotation=45,
               ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./networks/{platform}/output/domain_comparison_plot.png",
                dpi=300,
                bbox_inches='tight')
    plt.close()


def create_percentage_plot(platform, comparison_df):
    """
    Create and save a visualization of the top 5 domains by percentage (minimum 10 occurrences)
    """
    # Filter domains with at least 10 occurrences
    min_occurrences = 100 if platform == 'cross_platform' else 10
    filtered_domains = comparison_df[comparison_df['pre_filter'] >= min_occurrences].copy()

    # Calculate percentages for filtered domains
    filtered_domains['percentage'] = (filtered_domains['post_filter'] / filtered_domains['pre_filter'] * 100).round(2)

    # Get top 5 domains by percentage
    top_domains = filtered_domains.nlargest(5, 'percentage')

    # Sort by percentage from highest to lowest
    top_domains = top_domains.sort_values('percentage',
                                          ascending=True)  # ascending=True because we plot from bottom to top

    if len(top_domains) == 0:
        print("No domains found meeting the minimum occurrence threshold")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))  # Fixed size for 5 domains

    # Create horizontal bar chart
    bars = plt.barh(range(len(top_domains)), top_domains['percentage'], color='lightcoral')

    # Customize the plot
    plt.title(f'Top 5 Domains by Percentage in Filtered Content\n(minimum {min_occurrences} occurrences)')
    plt.xlabel('Percentage in filtered content')

    # Add domain names on y-axis
    plt.yticks(range(len(top_domains)), top_domains.index)

    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, i,
                 f"{width:.1f}% ({int(top_domains.iloc[i]['post_filter'])}/{int(top_domains.iloc[i]['pre_filter'])})",
                 va='center')

    # Adjust layout and save
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./networks/{platform}/output/top_percentage_domains.png",
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # Print statistics about these domains (in descending order)
    print(f"\nTop 5 domains by percentage (minimum {min_occurrences} occurrences):")
    for domain, row in top_domains[::-1].iterrows():  # Reverse order for printing
        print(f"{domain}: {row['percentage']:.1f}% ({int(row['post_filter'])}/{int(row['pre_filter'])} occurrences)")


def analyze_domains(platform):
    # Create output directories
    os.makedirs(f'./datasets/{platform}', exist_ok=True)
    os.makedirs(f'./networks/{platform}/output', exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_dataframes() if platform == 'cross_platform' else pd.read_csv(
        f"./datasets/{platform}/filtered_posts_cleaned.csv")
    author_names = load_authors(f"./networks/{platform}/output/network_authors.json")

    # Process domains
    print("Processing domains...")
    all_domain_counts = process_domains(df)
    filtered_domain_counts = process_domains(df, author_names)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'pre_filter': all_domain_counts,
        'post_filter': filtered_domain_counts
    }).fillna(0)

    # Create visualizations
    print("Creating visualizations...")
    create_comparison_plot(platform, comparison)
    create_percentage_plot(platform, comparison)

    # Print statistics
    total_posts_pre = comparison['pre_filter'].sum()
    total_posts_post = comparison['post_filter'].sum()
    print(f"\nGeneral statistics:")
    print(f"- Total links pre-filtering: {total_posts_pre:,.0f}")
    print(f"- Total links post-filtering: {total_posts_post:,.0f}")
    print(f"- Percentage of links from filtered authors: {(total_posts_post / total_posts_pre * 100):.2f}%")
    print(f"\nGenerated files:")
    print(f"- Volume comparison: './networks/{platform}/output/domain_comparison_plot.png'")
    print(f"- Percentage analysis: './networks/{platform}/output/top_percentage_domains.png'")


if __name__ == "__main__":
    analyze_domains('cross_platform')