"""
Data loading utilities for neuroimaging datasets.
Supports ABIDE, ADHD-200, and OASIS datasets from public repositories.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Import handling for optional neuroimaging packages with explicit None assignment
try:
    import nibabel as nib
    from nilearn import datasets
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    nib = None  # Explicit None assignment
    datasets = None  # Explicit None assignment
    print("Warning: Neuroimaging packages (nibabel, nilearn) not installed.")
    print("Run: pip install nibabel nilearn")

class DatasetLoader:
    """Base class for neuroimaging dataset loading."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> Dict[str, Any]:
        """Download dataset using nilearn's built-in downloaders."""
        raise NotImplementedError
        
    def load_phenotypic_data(self) -> pd.DataFrame:
        """Load phenotypic/demographic data."""
        raise NotImplementedError
        
    def get_subject_list(self) -> List[str]:
        """Get list of available subjects."""
        raise NotImplementedError

class ABIDELoader(DatasetLoader):
    """Loader for ABIDE (Autism Brain Imaging Data Exchange) dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        super().__init__(data_dir)
        self.dataset_dir = self.data_dir / "abide"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_dataset(self) -> Dict[str, Any]:
        """Download ABIDE dataset using nilearn."""
        if not NEUROIMAGING_AVAILABLE or datasets is None:
            print("Error: Neuroimaging packages required for data download.")
            print("Run: pip install nibabel nilearn")
            return {}
            
        print("Downloading ABIDE dataset...")
        
        try:
            # Use nilearn's ABIDE dataset fetcher
            abide_data = datasets.fetch_abide_pcp(
                data_dir=str(self.dataset_dir),
                pipeline='cpac',
                band_pass_filtering=True,
                global_signal_regression=False,
                derivatives=['func_preproc'],
                quality_checked=True,
                n_subjects=30  # Limit for rapid development
            )
            
            # Reorganize downloaded files into expected structure
            neuroimaging_dir = self.dataset_dir / "neuroimaging"
            phenotypic_dir = self.dataset_dir / "phenotypic"
            neuroimaging_dir.mkdir(exist_ok=True)
            phenotypic_dir.mkdir(exist_ok=True)

            # Move functional data
            if hasattr(abide_data, 'func_preproc') and abide_data.func_preproc:
                print("Organizing ABIDE functional data...")
                for func_file in abide_data.func_preproc:
                    src_path = Path(func_file)
                    dest_path = neuroimaging_dir / src_path.name
                    if not dest_path.exists(): # Check if destination exists
                        try:
                            src_path.rename(dest_path)
                        except FileNotFoundError:
                            print(f"Warning: Functional file not found at {src_path}")
                        except Exception as e:
                            print(f"Error moving functional file {src_path}: {e}")
                    else:
                        print(f"Functional file already exists at {dest_path}. Skipping move.")


            # Organize phenotypic data
            phenotypic_file_path = None
            if hasattr(abide_data, 'phenotypic'):
                 print("Organizing ABIDE phenotypic data...")
                 phenotypic_dir.mkdir(exist_ok=True)
                 if isinstance(abide_data.phenotypic, str) and Path(abide_data.phenotypic).exists():
                     # If it's a path string, move the file
                     src_path = Path(abide_data.phenotypic)
                     phenotypic_file_path = phenotypic_dir / src_path.name
                     if not phenotypic_file_path.exists(): # Check if destination exists
                         try:
                             src_path.rename(phenotypic_file_path)
                         except Exception as e:
                             print(f"Error moving phenotypic file {src_path}: {e}")
                             phenotypic_file_path = None # Ensure it's None if move fails
                         else:
                             print(f"Phenotypic file already exists at {phenotypic_file_path}. Skipping move.")
                 elif isinstance(abide_data.phenotypic, pd.DataFrame) and not abide_data.phenotypic.empty:
                     # If it's a DataFrame, save it to CSV
                     try:
                         phenotypic_file_name = "ABIDE_pcp_phenotypic.csv" # Define a default filename
                         phenotypic_file_path = phenotypic_dir / phenotypic_file_name
                         if not phenotypic_file_path.exists(): # Check if destination exists
                             abide_data.phenotypic.to_csv(phenotypic_file_path, index=False)
                         else:
                             print(f"Phenotypic file already exists at {phenotypic_file_path}. Skipping save.")
                     except Exception as e:
                         print(f"Error saving phenotypic DataFrame to CSV: {e}")
                         phenotypic_file_path = None # Ensure it's None if save fails
                 else:
                     print("Warning: ABIDE phenotypic data not found or is empty after download.")
            else:
                 print("Warning: ABIDE data object has no 'phenotypic' attribute.")


            return {
                'functional_data': list(neuroimaging_dir.glob('*.nii*')), # Return new paths
                'phenotypic': phenotypic_file_path, # Return the path to the organized phenotypic file
                'description': abide_data.description
            }
        except Exception as e:
            print(f"Error downloading or organizing ABIDE data: {e}")
            return {}
    
    def load_phenotypic_data(self) -> pd.DataFrame:
        """Load ABIDE phenotypic data."""
        # Look for the phenotypic file in the expected location
        phenotypic_path = self.dataset_dir / "phenotypic" / "ABIDE_pcp_phenotypic.csv" # Assuming the filename used in download_dataset
        if phenotypic_path.exists():
            try:
                return pd.read_csv(phenotypic_path)
            except Exception as e:
                print(f"Error reading ABIDE phenotypic data from {phenotypic_path}: {e}")
                return pd.DataFrame()
        else:
            print(f"ABIDE phenotypic file not found at {phenotypic_path}.")
            return pd.DataFrame()

# class ADHD200Loader(DatasetLoader):
#     """Loader for ADHD-200 dataset."""
    
#     def __init__(self, data_dir: str = "data/raw"):
#         super().__init__(data_dir)
#         self.dataset_dir = self.data_dir / "adhd200"
#         self.dataset_dir.mkdir(exist_ok=True)
        
#     def download_dataset(self) -> Dict[str, Any]:
#         """Download ADHD-200 dataset using nilearn."""
#         if not NEUROIMAGING_AVAILABLE or datasets is None:
#             print("Error: Neuroimaging packages required for data download.")
#             print("Run: pip install nibabel nilearn")
#             return {}
            
#         print("Downloading ADHD-200 dataset...")
        
#         try:
#             adhd_data = datasets.fetch_adhd(
#                 data_dir=str(self.dataset_dir),
#                 n_subjects=20  # Start with subset for rapid development
#             )
            
#             # Reorganize downloaded files into expected structure
#             neuroimaging_dir = self.dataset_dir / "neuroimaging"
#             phenotypic_dir = self.dataset_dir / "phenotypic"
#             neuroimaging_dir.mkdir(exist_ok=True)
#             phenotypic_dir.mkdir(exist_ok=True)

#             # Organize functional data
#             if hasattr(adhd_data, 'func') and adhd_data.func is not None:
#                 functional_files = list(adhd_data.func) # Ensure it's a list
#                 if len(functional_files) > 0:
#                     print("Organizing ADHD-200 functional data...")
#                     for func_file in functional_files:
#                         print(f"Processing functional file: Type={type(func_file)}, Value={func_file}") # Added print statement
#                         src_path = Path(func_file)
#                         dest_path = neuroimaging_dir / src_path.name
#                         if not dest_path.exists(): # Check if destination exists
#                             try:
#                                 src_path.rename(dest_path)
#                             except FileNotFoundError:
#                                 print(f"Warning: Functional file not found at {src_path}")
#                             except Exception as e:
#                                 print(f"Error moving functional file {src_path}: {e}")
#                         else:
#                             print(f"Functional file already exists at {dest_path}. Skipping move.")
#                 else:
#                     print("Warning: ADHD-200 functional data list is empty after download.")
#             else:
#                 print("Warning: ADHD-200 data object has no 'func' attribute or it is None.")


#             # Organize phenotypic data
#             phenotypic_file_path = None
#             if hasattr(adhd_data, 'phenotypic') and adhd_data.phenotypic is not None:
#                  print("Organizing ADHD-200 phenotypic data...")
#                  phenotypic_dir.mkdir(exist_ok=True)
#                  if isinstance(adhd_data.phenotypic, str) and Path(adhd_data.phenotypic).exists():
#                      # If it's a path string, move the file
#                      src_path = Path(adhd_data.phenotypic)
#                      phenotypic_file_path = phenotypic_dir / src_path.name
#                      if not phenotypic_file_path.exists(): # Check if destination exists
#                          try:
#                              src_path.rename(phenotypic_file_path)
#                          except Exception as e:
#                              print(f"Error moving phenotypic file {src_path}: {e}")
#                              phenotypic_file_path = None # Ensure it's None if move fails
#                          else:
#                              print(f"Phenotypic file already exists at {phenotypic_file_path}. Skipping move.")
#                  # No need to handle DataFrame for ADHD-200 phenotypic based on typical nilearn behavior
#                  else:
#                      print("Warning: ADHD-200 phenotypic data is not a valid path string or does not exist after download.")
#             else:
#                  print("Warning: ADHD-200 data object has no 'phenotypic' attribute or it is None.")


#             confounds_data = getattr(adhd_data, 'confounds', None)
#             if isinstance(confounds_data, np.ndarray):
#                 # If confounds is a numpy array, convert it to a list to avoid potential ambiguity issues
#                 confounds_data = confounds_data.tolist()

#             # Explicitly delete the adhd_data object to prevent potential lingering issues
#             del adhd_data

#             return {
#                 'functional_data': list(neuroimaging_dir.glob('*.nii*')), # Return new paths
#                 'phenotypic': phenotypic_file_path, # Return new path
#                 'confounds': confounds_data # Use the potentially converted confounds data
#             }
#         except Exception as e:
#             print(f"Error downloading or organizing ADHD-200 data: {e}")
#             return {}

#     def load_phenotypic_data(self) -> pd.DataFrame:
#         """Load ADHD-200 phenotypic data."""
#         phenotypic_path = self.dataset_dir / "phenotypic" / "ADHD200_a_symp_qc.csv" # Assuming default nilearn filename
#         if phenotypic_path.exists():
#             try:
#                 return pd.read_csv(phenotypic_path)
#             except Exception as e:
#                 print(f"Error reading ADHD-200 phenotypic data from {phenotypic_path}: {e}")
#                 return pd.DataFrame()
#         else:
#             print(f"ADHD-200 phenotypic file not found at {phenotypic_path}.")
#             return pd.DataFrame()


def get_available_datasets() -> List[str]:
    """Return list of available dataset loaders."""
    return ['abide']

def create_dataset_loader(dataset_name: str, data_dir: str = "data/raw") -> DatasetLoader:
    """Factory function to create appropriate dataset loader."""
    loaders = {
        'abide': ABIDELoader,
        # 'adhd200': ADHD200Loader # Commented out
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()](data_dir)
