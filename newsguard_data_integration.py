import pandas as pd
import os
from pathlib import Path


def merge_domain_data(platform):
    """
    Merge domain evaluation data with domain occurrences for a specific platform.

    Args:
        platform (str): Name of the platform folder containing domain_occurrences.csv

    Returns:
        pd.DataFrame: Merged dataframe containing combined information
    """
    # Read the domain evaluation data
    domain_eval_path = Path('./datasets/newsguard/domains_evaluation.csv')
    domain_eval_df = pd.read_csv(domain_eval_path)

    # Read the platform-specific domain occurrences
    occurrences_path = Path(f'./datasets/{platform}/domain_occurrences.csv')
    occurrences_df = pd.read_csv(occurrences_path)

    # Merge the dataframes on domain columns
    # Assuming the first column in occurrences_df is the domain column
    merged_df = pd.merge(
        occurrences_df,
        domain_eval_df,
        left_on=occurrences_df.columns[0],  # First column as merge key
        right_on='Domain',
        how='left'  # Keep all records from occurrences_df
    )

    # Save the merged dataframe
    output_dir = Path(f'./datasets/{platform}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'merged_domain_data.csv'
    merged_df.to_csv(output_path, index=False)

    print(f"Merged data saved to: {output_path}")
    return merged_df


def process_all_platforms():
    """
    Process all platform folders in the datasets directory that contain domain_occurrences.csv
    """
    datasets_dir = Path('./datasets')

    # Get all platform folders (excluding newsguard)
    platforms = [
        d.name for d in datasets_dir.iterdir()
        if d.is_dir() and d.name != 'newsguard'
           and (d / 'domain_occurrences.csv').exists()
    ]

    results = {}
    for platform in platforms:
        print(f"\nProcessing platform: {platform}")
        try:
            results[platform] = merge_domain_data(platform)
            print(f"Successfully processed {platform}")
        except Exception as e:
            print(f"Error processing {platform}: {str(e)}")

    return results


if __name__ == "__main__":
    # Process all platforms
    results = process_all_platforms()

    # Print summary
    print("\nSummary of processed data:")
    for platform, df in results.items():
        print(f"\n{platform}:")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")