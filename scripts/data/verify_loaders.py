import sys
from pathlib import Path

# Add the src directory to the Python path to import loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from connectome_analysis.data.loaders import (
        ABIDELoader,
        ADHD200Loader,
        create_dataset_loader,
        NEUROIMAGING_AVAILABLE,
    )
    print("Successfully imported loader classes and factory function.")

    # 1. Check NEUROIMAGING_AVAILABLE flag
    if NEUROIMAGING_AVAILABLE:
        print("Neuroimaging packages (nibabel, nilearn) are available.")
    else:
        print("Neuroimaging packages (nibabel, nilearn) are NOT available.")
        print("Data download and full loading functionality may be limited.")

    # 2. Test create_dataset_loader factory function
    print("\nTesting create_dataset_loader factory function...")
    abide_loader_instance = None
    adhd200_loader_instance = None
    try:
        abide_loader_instance = create_dataset_loader("abide")
        print(f"create_dataset_loader('abide') returned: {type(abide_loader_instance)}")
        assert isinstance(abide_loader_instance, ABIDELoader)
        print("create_dataset_loader('abide') test passed.")
    except Exception as e:
        print(f"create_dataset_loader('abide') test failed: {e}")

    try:
        adhd200_loader_instance = create_dataset_loader("adhd200")
        print(f"create_dataset_loader('adhd200') returned: {type(adhd200_loader_instance)}")
        assert isinstance(adhd200_loader_instance, ADHD200Loader)
        print("create_dataset_loader('adhd200') test passed.")
    except Exception as e:
        print(f"create_dataset_loader('adhd200') test failed: {e}")

    try:
        create_dataset_loader("invalid_dataset")
        print("create_dataset_loader('invalid_dataset') test failed: ValueError not raised.")
    except ValueError as e:
        print(f"create_dataset_loader('invalid_dataset') test passed: ValueError raised - {e}")
    except Exception as e:
        print(f"create_dataset_loader('invalid_dataset') test failed: Unexpected error - {e}")

    # 3. Check for expected methods in loader instances (basic check without data)
    print("\nChecking for expected methods in loader instances...")
    if 'abide_loader_instance' in locals():
        print(f"ABIDELoader has download_dataset method: {hasattr(abide_loader_instance, 'download_dataset')}")
        print(f"ABIDELoader has load_phenotypic_data method: {hasattr(abide_loader_instance, 'load_phenotypic_data')}")
        print(f"ABIDELoader has get_subject_list method: {hasattr(abide_loader_instance, 'get_subject_list')}")
    
    if 'adhd200_loader_instance' in locals():
        print(f"ADHD200Loader has download_dataset method: {hasattr(adhd200_loader_instance, 'download_dataset')}")
        print(f"ADHD200Loader has load_phenotypic_data method: {hasattr(adhd200_loader_instance, 'load_phenotypic_data')}")
        print(f"ADHD200Loader has get_subject_list method: {hasattr(adhd200_loader_instance, 'get_subject_list')}")

except ImportError as e:
    print(f"Error importing loader classes: {e}")
    print("Please ensure the 'src' directory is correctly structured and dependencies are installed.")

print("\nLoader verification script finished.")
