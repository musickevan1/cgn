import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the Python path to import loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from connectome_analysis.data.loaders import ABIDELoader

def download_dataset(dataset_name, data_dir):
    """Downloads the specified dataset using nilearn."""
    print(f"Attempting to download {dataset_name} dataset...")
    try:
        if dataset_name == 'abide':
            # Use ABIDELoader to get the nilearn fetcher and download
            loader = ABIDELoader(data_dir)
            loader.download_dataset()
        else:
            # This case should ideally not be reached with the updated argument parser
            print(f"Error: Only 'abide' dataset is supported at this time.")
            return False
        print(f"Successfully downloaded {dataset_name} dataset.")
        return True
    except Exception as e:
        print(f"Error downloading {dataset_name} dataset: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the ABIDE neuroimaging dataset for CGN project.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["abide"],
        default="abide", # Set default to abide
        help="Specify which dataset to download: 'abide'. Defaults to 'abide'."
    )

    args = parser.parse_args()

    # Define the base data directory
    base_data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    os.makedirs(base_data_dir, exist_ok=True)

    # The logic is simplified to only download abide
    download_dataset("abide", base_data_dir)

    print("Dataset download process finished.")
