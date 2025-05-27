import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path to import loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from connectome_analysis.data.loaders import create_dataset_loader, NEUROIMAGING_AVAILABLE
    if NEUROIMAGING_AVAILABLE:
        import nibabel as nib
    else:
        nib = None
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure the 'src' directory is correctly structured and dependencies are installed.")
    sys.exit(1)

def validate_dataset(dataset_name, data_dir):
    """Validates the downloaded dataset."""
    print(f"Validating {dataset_name} dataset...")
    
    dataset_path = Path(data_dir) / dataset_name
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found at {dataset_path}")
        return False
        
    # Check for expected subdirectories
    neuroimaging_path = dataset_path / "neuroimaging"
    phenotypic_path = dataset_path / "phenotypic"
    
    print(f"Checking for neuroimaging directory: {neuroimaging_path.exists()}")
    print(f"Checking for phenotypic directory: {phenotypic_path.exists()}")

    is_valid = True

    # Validate phenotypic data
    print("\nValidating phenotypic data...")
    try:
        loader = create_dataset_loader(dataset_name, data_dir)
        phenotypic_data = loader.load_phenotypic_data()
        if not phenotypic_data.empty:
            print(f"Phenotypic data loaded successfully. Shape: {phenotypic_data.shape}")
            # TODO: Add more specific checks for expected columns and data types
        else:
            print("Warning: Phenotypic data not found or is empty.")
            is_valid = False # Consider this a warning, not a hard failure for now
    except Exception as e:
        print(f"Error loading phenotypic data: {e}")
        is_valid = False

    # Validate neuroimaging data (basic check for file existence)
    print("\nValidating neuroimaging data...")
    if neuroimaging_path.exists():
        nii_files = list(neuroimaging_path.glob('*.nii')) + list(neuroimaging_path.glob('*.nii.gz'))
        if nii_files:
            print(f"Found {len(nii_files)} neuroimaging files.")
            # TODO: Add more detailed checks using nibabel if NEUROIMAGING_AVAILABLE
            if NEUROIMAGING_AVAILABLE and nib:
                 print("Neuroimaging packages available for detailed file checks (not yet implemented).")
            else:
                 print("Neuroimaging packages not available. Skipping detailed file checks.")
        else:
            print("Warning: No .nii or .nii.gz files found in neuroimaging directory.")
            is_valid = False # Consider this a warning
    else:
        print("Neuroimaging directory not found. Skipping neuroimaging data validation.")
        is_valid = False # Consider this a warning

    # TODO: Add dataset statistics reporting (number of subjects, sites, etc.)

    return is_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate neuroimaging datasets for CGN project.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["abide", "adhd200", "all"],
        required=True,
        help="Specify which dataset to validate: 'abide', 'adhd200', or 'all'."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Base directory where datasets are stored."
    )

    args = parser.parse_args()

    base_data_dir = Path(args.data_dir)

    if args.dataset == "all":
        abide_valid = validate_dataset("abide", base_data_dir)
        adhd200_valid = validate_dataset("adhd200", base_data_dir)
        if abide_valid and adhd200_valid:
            print("\nAll specified datasets validated successfully (with warnings noted).")
        else:
            print("\nValidation finished with issues for one or more datasets.")
    else:
        is_valid = validate_dataset(args.dataset, base_data_dir)
        if is_valid:
            print(f"\n{args.dataset} dataset validated successfully (with warnings noted).")
        else:
            print(f"\nValidation finished with issues for {args.dataset} dataset.")

    print("Dataset validation process finished.")
