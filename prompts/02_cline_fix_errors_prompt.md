# Cline Fix Prompt - Resolve Code Errors & Dependencies

## Current Issues
The CGN project has several Pylance errors that need to be fixed. These fall into categories:
1. Missing package dependencies 
2. Method signature mismatches
3. Type annotation issues
4. Import path corrections

## Task 1: Fix Method Signatures in loaders.py

### Replace the entire `src/connectome_analysis/data/loaders.py` file with this corrected version:

```python
"""
Data loading utilities for neuroimaging datasets.
Supports ABIDE, ADHD-200, and OASIS datasets from public repositories.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import handling for optional neuroimaging packages
try:
    import nibabel as nib
    from nilearn import datasets
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    print("Warning: Neuroimaging packages (nibabel, nilearn) not installed.")
    print("Run: pip install nibabel nilearn")

class DatasetLoader:
    """Base class for neuroimaging dataset loading."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> Dict:  # Fixed: removed dataset_name parameter
        """Download dataset using nilearn's built-in downloaders."""
        raise NotImplementedError
        
    def load_phenotypic_data(self) -> pd.DataFrame:  # Fixed: removed dataset_name parameter
        """Load phenotypic/demographic data."""
        raise NotImplementedError
        
    def get_subject_list(self) -> List[str]:  # Fixed: removed dataset_name parameter
        """Get list of available subjects."""
        raise NotImplementedError

class ABIDELoader(DatasetLoader):
    """Loader for ABIDE (Autism Brain Imaging Data Exchange) dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        super().__init__(data_dir)
        self.dataset_dir = self.data_dir / "abide"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_dataset(self) -> Dict:  # Fixed: matches parent signature
        """Download ABIDE dataset using nilearn."""
        if not NEUROIMAGING_AVAILABLE:
            print("Error: Neuroimaging packages required for data download.")
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
            
            return {
                'functional_data': abide_data.func_preproc,
                'phenotypic': abide_data.phenotypic,
                'description': abide_data.description
            }
        except Exception as e:
            print(f"Error downloading ABIDE data: {e}")
            return {}
    
    def load_phenotypic_data(self) -> pd.DataFrame:  # Fixed: matches parent signature
        """Load ABIDE phenotypic data."""
        # First try to download if not exists
        try:
            abide_data = self.download_dataset()
            if 'phenotypic' in abide_data:
                return abide_data['phenotypic']
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading ABIDE data: {e}")
            return pd.DataFrame()

class ADHD200Loader(DatasetLoader):
    """Loader for ADHD-200 dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        super().__init__(data_dir)
        self.dataset_dir = self.data_dir / "adhd200"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_dataset(self) -> Dict:  # Fixed: matches parent signature
        """Download ADHD-200 dataset using nilearn."""
        if not NEUROIMAGING_AVAILABLE:
            print("Error: Neuroimaging packages required for data download.")
            return {}
            
        print("Downloading ADHD-200 dataset...")
        
        try:
            adhd_data = datasets.fetch_adhd(
                data_dir=str(self.dataset_dir),
                n_subjects=20  # Start with subset for rapid development
            )
            
            return {
                'functional_data': adhd_data.func,
                'phenotypic': adhd_data.phenotypic,
                'confounds': adhd_data.confounds if hasattr(adhd_data, 'confounds') else None
            }
        except Exception as e:
            print(f"Error downloading ADHD-200 data: {e}")
            return {}

def get_available_datasets() -> List[str]:
    """Return list of available dataset loaders."""
    return ['abide', 'adhd200']

def create_dataset_loader(dataset_name: str, data_dir: str = "data/raw") -> DatasetLoader:
    """Factory function to create appropriate dataset loader."""
    loaders = {
        'abide': ABIDELoader,
        'adhd200': ADHD200Loader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()](data_dir)
```

## Task 2: Fix download_datasets.py Script

### Replace `scripts/data/download_datasets.py` with this corrected version:

```python
#!/usr/bin/env python3
"""
Download script for neuroimaging datasets.
Run this to get the data needed for the CGN project.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / 'src'))

from connectome_analysis.data.loaders import create_dataset_loader, get_available_datasets

def main():
    parser = argparse.ArgumentParser(description='Download neuroimaging datasets')
    parser.add_argument('--dataset', choices=get_available_datasets(), 
                       default='abide', help='Dataset to download')
    parser.add_argument('--data-dir', default='data/raw', 
                       help='Directory to store downloaded data')
    parser.add_argument('--n-subjects', type=int, default=50,
                       help='Number of subjects to download (for rapid prototyping)')
    
    args = parser.parse_args()
    
    print(f"CGN Data Downloader")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max subjects: {args.n_subjects}")
    print("-" * 40)
    
    # Create data loader
    loader = create_dataset_loader(args.dataset, args.data_dir)  # Fixed: added missing argument
    
    # Download dataset
    try:
        data = loader.download_dataset()  # Fixed: no argument needed
        print(f"✅ Successfully downloaded {args.dataset} dataset")
        
        if 'phenotypic' in data and data['phenotypic'] is not None:
            print(f"📊 Phenotypic data shape: {data['phenotypic'].shape}")
        if 'functional_data' in data and data['functional_data'] is not None:
            print(f"🧠 Functional data files: {len(data['functional_data'])}")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("This might be due to network issues or data server problems.")
        print("Try again later or check the dataset documentation.")
        return 1
    
    print("✅ Data download complete!")
    return 0

if __name__ == '__main__':
    exit(main())
```

## Task 3: Fix preprocessing.py Type Issues

### Replace the problematic sections in `src/connectome_analysis/data/preprocessing.py`:

```python
"""
Preprocessing utilities for neuroimaging data.
Handles basic preprocessing, quality control, and standardization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
import warnings

# Import handling for optional neuroimaging packages
try:
    import nibabel as nib
    from nilearn import image, masking, signal
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker  # Updated import path
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    print("Warning: Neuroimaging packages not installed.")
    print("Run: pip install nibabel nilearn")

class NeuroPreprocessor:
    """Main preprocessing class for neuroimaging data."""
    
    def __init__(self, 
                 standardize: bool = True,
                 detrend: bool = True,
                 low_pass: Optional[float] = 0.1,
                 high_pass: Optional[float] = 0.01,
                 t_r: float = 2.0):
        """
        Initialize preprocessor with common parameters.
        
        Parameters:
        -----------
        standardize : bool
            Whether to standardize time series
        detrend : bool
            Whether to remove linear trends
        low_pass : float or None
            Low-pass filter frequency in Hz
        high_pass : float or None
            High-pass filter frequency in Hz
        t_r : float
            Repetition time in seconds
        """
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        
    def preprocess_fmri(self, 
                       fmri_img,  # Removed type hint to avoid nibabel dependency
                       mask_img=None) -> np.ndarray:
        """
        Preprocess fMRI data with standard pipeline.
        
        Parameters:
        -----------
        fmri_img : nibabel image
            4D fMRI image
        mask_img : nibabel image, optional
            Brain mask image
            
        Returns:
        --------
        time_series : ndarray
            Preprocessed time series data
        """
        
        if not NEUROIMAGING_AVAILABLE:
            raise ImportError("Neuroimaging packages required for preprocessing")
        
        # Apply brain mask if provided
        if mask_img is not None:
            time_series = masking.apply_mask(fmri_img, mask_img)
        else:
            # Create mask from data
            mask_img = masking.compute_epi_mask(fmri_img)
            time_series = masking.apply_mask(fmri_img, mask_img)
        
        # Apply temporal filtering and standardization
        time_series = signal.clean(
            time_series,
            detrend=self.detrend,
            standardize=self.standardize,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r
        )
        
        return time_series
    
    def extract_roi_time_series(self, 
                               fmri_img,  # Removed type hint
                               atlas_img) -> np.ndarray:  # Removed type hint
        """
        Extract ROI time series using an atlas.
        
        Parameters:
        -----------
        fmri_img : nibabel image
            4D fMRI image
        atlas_img : nibabel image
            3D atlas image with integer labels
            
        Returns:
        --------
        roi_time_series : ndarray
            Time series for each ROI (n_timepoints, n_rois)
        """
        
        if not NEUROIMAGING_AVAILABLE:
            raise ImportError("Neuroimaging packages required for ROI extraction")
        
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=self.standardize,
            detrend=self.detrend,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r,
            verbose=0
        )
        
        roi_time_series = masker.fit_transform(fmri_img)
        return roi_time_series

class QualityControl:
    """Quality control utilities for neuroimaging data."""
    
    @staticmethod
    def compute_motion_metrics(confounds: pd.DataFrame) -> Dict[str, float]:
        """Compute motion-related quality metrics."""
        motion_columns = [col for col in confounds.columns 
                         if any(motion_term in col.lower() 
                               for motion_term in ['trans', 'rot', 'motion'])]
        
        if not motion_columns:
            return {'mean_fd': np.nan, 'max_fd': np.nan, 'n_high_motion': 0}
        
        # Compute framewise displacement if available
        if 'framewise_displacement' in confounds.columns:
            fd = confounds['framewise_displacement'].values
        else:
            # Approximate FD from motion parameters
            motion_params = confounds[motion_columns].values
            fd = np.sum(np.abs(np.diff(motion_params, axis=0)), axis=1)
        
        # Fixed: Convert to numpy arrays and handle types properly
        fd_array = np.array(fd, dtype=float)
        
        return {
            'mean_fd': float(np.nanmean(fd_array)),
            'max_fd': float(np.nanmax(fd_array)),
            'n_high_motion': int(np.sum(fd_array > 0.5))  # Frames with FD > 0.5mm
        }
    
    @staticmethod
    def compute_signal_metrics(time_series: np.ndarray) -> Dict[str, float]:
        """Compute signal quality metrics."""
        # Temporal signal-to-noise ratio
        tsnr = np.mean(time_series, axis=0) / np.std(time_series, axis=0)
        
        # Global signal properties
        global_signal = np.mean(time_series, axis=1)
        
        return {
            'mean_tsnr': float(np.nanmean(tsnr)),
            'global_signal_std': float(np.std(global_signal)),
            'n_outlier_volumes': int(np.sum(np.abs(global_signal - np.mean(global_signal)) > 3 * np.std(global_signal)))
        }

def load_standard_atlases() -> Dict[str, str]:
    """Download and return paths to standard brain atlases."""
    if not NEUROIMAGING_AVAILABLE:
        print("Warning: Neuroimaging packages not available for atlas download")
        return {}
    
    from nilearn import datasets
    
    atlases = {}
    
    # AAL atlas
    try:
        aal = datasets.fetch_atlas_aal(version='SPM12')
        atlases['aal'] = aal.maps
    except Exception as e:
        print(f"Could not download AAL atlas: {e}")
    
    # Schaefer atlas
    try:
        schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        atlases['schaefer_400'] = schaefer.maps
    except Exception as e:
        print(f"Could not download Schaefer atlas: {e}")
    
    return atlases
```

## Task 4: Fix connectome.py NetworkX Community Detection

### Replace the community detection section in `src/connectome_analysis/data/connectome.py`:

```python
"""
Connectome construction utilities.
Build functional and structural connectivity matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
import warnings

# Import handling for optional packages
try:
    from nilearn.connectome import ConnectivityMeasure
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available. Limited connectivity options.")

class ConnectomeBuilder:
    """Build connectome matrices from neuroimaging data."""
    
    def __init__(self, 
                 connectivity_kinds: List[str] = ['correlation', 'partial correlation'],
                 standardize_connectomes: bool = True):
        """
        Initialize connectome builder.
        
        Parameters:
        -----------
        connectivity_kinds : list
            Types of connectivity to compute
        standardize_connectomes : bool
            Whether to standardize connectivity matrices
        """
        self.connectivity_kinds = connectivity_kinds
        self.standardize_connectomes = standardize_connectomes
        
        # Initialize connectivity measures
        self.connectivity_measures = {}
        if NILEARN_AVAILABLE:
            for kind in connectivity_kinds:
                self.connectivity_measures[kind] = ConnectivityMeasure(
                    kind=kind,
                    standardize=standardize_connectomes
                )
        else:
            print("Warning: Using basic correlation only (nilearn not available)")
    
    def build_functional_connectome(self, 
                                  roi_time_series: np.ndarray,
                                  subject_id: str) -> Dict[str, np.ndarray]:
        """
        Build functional connectivity matrices.
        
        Parameters:
        -----------
        roi_time_series : ndarray
            ROI time series (n_timepoints, n_rois)
        subject_id : str
            Subject identifier
            
        Returns:
        --------
        connectomes : dict
            Dictionary containing different connectivity matrices
        """
        connectomes = {}
        
        if NILEARN_AVAILABLE:
            for kind, measure in self.connectivity_measures.items():
                try:
                    # Fit and transform time series
                    connectivity_matrix = measure.fit_transform([roi_time_series])[0]
                    connectomes[kind] = connectivity_matrix
                    
                except Exception as e:
                    print(f"Warning: Could not compute {kind} connectivity for {subject_id}: {e}")
                    connectomes[kind] = np.full((roi_time_series.shape[1], roi_time_series.shape[1]), np.nan)
        else:
            # Fallback to basic correlation
            try:
                correlation_matrix = np.corrcoef(roi_time_series.T)
                connectomes['correlation'] = correlation_matrix
            except Exception as e:
                print(f"Warning: Could not compute correlation for {subject_id}: {e}")
                connectomes['correlation'] = np.full((roi_time_series.shape[1], roi_time_series.shape[1]), np.nan)
        
        return connectomes
    
    def compute_graph_metrics(self, 
                             connectivity_matrix: np.ndarray,
                             threshold: Optional[float] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute graph theory metrics from connectivity matrix.
        
        Parameters:
        -----------
        connectivity_matrix : ndarray
            Connectivity matrix
        threshold : float, optional
            Threshold for binarizing the matrix
            
        Returns:
        --------
        metrics : dict
            Dictionary of graph metrics
        """
        
        # Handle NaN values
        if np.isnan(connectivity_matrix).any():
            return {metric: np.nan for metric in 
                   ['global_efficiency', 'local_efficiency', 'modularity', 
                    'clustering', 'path_length', 'small_worldness']}
        
        # Apply threshold if specified
        if threshold is not None:
            adj_matrix = np.abs(connectivity_matrix) > threshold
        else:
            adj_matrix = np.abs(connectivity_matrix)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix)
        
        try:
            # Global metrics
            if threshold is not None:  # Binary graph
                global_efficiency = nx.global_efficiency(G)
                local_efficiency = nx.local_efficiency(G)
                clustering = nx.average_clustering(G)
                try:
                    path_length = nx.average_shortest_path_length(G)
                except:
                    path_length = np.nan
            else:  # Weighted graph
                global_efficiency = nx.global_efficiency(G)
                local_efficiency = nx.local_efficiency(G)
                clustering = nx.average_clustering(G, weight='weight')
                path_length = np.nan  # Not well-defined for weighted graphs
            
            # Modularity - Fixed community detection
            try:
                # Use updated NetworkX community detection
                communities = nx.community.louvain_communities(G)  # Fixed: use louvain_communities
                modularity = nx.community.modularity(G, communities)
            except Exception as e:
                try:
                    # Fallback to greedy modularity (if available)
                    communities = list(nx.community.greedy_modularity_communities(G))
                    modularity = nx.community.modularity(G, communities)
                except:
                    modularity = np.nan
            
            # Small-worldness
            if not np.isnan(clustering) and not np.isnan(path_length):
                # Generate random graph for comparison
                n_nodes = len(G.nodes())
                n_edges = len(G.edges())
                if n_nodes > 1 and n_edges > 0:
                    p = n_edges / (n_nodes * (n_nodes - 1) / 2)
                    random_G = nx.erdos_renyi_graph(n_nodes, p)
                    
                    random_clustering = nx.average_clustering(random_G)
                    try:
                        random_path_length = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else np.nan
                    except:
                        random_path_length = np.nan
                    
                    if not np.isnan(random_clustering) and not np.isnan(random_path_length) and random_clustering > 0 and random_path_length > 0:
                        small_worldness = (clustering / random_clustering) / (path_length / random_path_length)
                    else:
                        small_worldness = np.nan
                else:
                    small_worldness = np.nan
            else:
                small_worldness = np.nan
            
            return {
                'global_efficiency': global_efficiency,
                'local_efficiency': local_efficiency,
                'modularity': modularity,
                'clustering': clustering,
                'path_length': path_length,
                'small_worldness': small_worldness
            }
            
        except Exception as e:
            print(f"Warning: Could not compute graph metrics: {e}")
            return {metric: np.nan for metric in 
                   ['global_efficiency', 'local_efficiency', 'modularity', 
                    'clustering', 'path_length', 'small_worldness']}

class MultiModalConnectome:
    """Build multi-modal connectomes combining different data types."""
    
    def __init__(self):
        self.functional_builder = ConnectomeBuilder()
        
    def combine_modalities(self, 
                          functional_connectome: np.ndarray,
                          structural_connectome: Optional[np.ndarray] = None,
                          method: str = 'concatenate') -> np.ndarray:
        """
        Combine functional and structural connectivity information.
        
        Parameters:
        -----------
        functional_connectome : ndarray
            Functional connectivity matrix
        structural_connectome : ndarray, optional
            Structural connectivity matrix
        method : str
            Method for combining modalities ('concatenate', 'multiply', 'average')
            
        Returns:
        --------
        combined_connectome : ndarray
            Combined connectivity representation
        """
        
        if structural_connectome is None:
            return functional_connectome
        
        if method == 'concatenate':
            # Stack upper triangular parts
            func_triu = functional_connectome[np.triu_indices_from(functional_connectome, k=1)]
            struct_triu = structural_connectome[np.triu_indices_from(structural_connectome, k=1)]
            return np.concatenate([func_triu, struct_triu])
        
        elif method == 'multiply':
            # Element-wise multiplication
            return functional_connectome * structural_connectome
        
        elif method == 'average':
            # Average of the two matrices
            return (functional_connectome + structural_connectome) / 2
        
        else:
            raise ValueError(f"Unknown combination method: {method}")

def extract_connectome_features(connectome: np.ndarray) -> np.ndarray:
    """Extract feature vector from connectivity matrix."""
    # Use upper triangular part (excluding diagonal)
    triu_indices = np.triu_indices_from(connectome, k=1)
    features = connectome[triu_indices]
    
    # Remove NaN values
    features = features[~np.isnan(features)]
    
    return features
```

## Task 5: Create a Safe Installation Script

### Create `scripts/utils/install_dependencies.py`:

```python
#!/usr/bin/env python3
"""
Install script for CGN project dependencies.
Handles optional neuroimaging packages gracefully.
"""

import subprocess
import sys
from pathlib import Path

def install_package(package_name, optional=False):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"⚠️  Optional package {package_name} failed to install: {e}")
            print("   This may be due to system dependencies. Project will work with limited functionality.")
            return False
        else:
            print(f"❌ Failed to install required package {package_name}: {e}")
            return False

def main():
    print("CGN Dependencies Installer")
    print("=" * 30)
    
    # Required packages
    required_packages = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0"
    ]
    
    # Optional neuroimaging packages
    optional_packages = [
        "nibabel>=5.0.0",
        "nilearn>=0.10.0",
        "torch-geometric>=2.3.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0"
    ]
    
    print("Installing required packages...")
    failed_required = []
    for package in required_packages:
        if not install_package(package):
            failed_required.append(package)
    
    print("\nInstalling optional packages...")
    failed_optional = []
    for package in optional_packages:
        if not install_package(package, optional=True):
            failed_optional.append(package)
    
    print("\n" + "=" * 30)
    print("INSTALLATION SUMMARY")
    print("=" * 30)
    
    if not failed_required:
        print("✅ All required packages installed successfully!")
    else:
        print(f"❌ Failed required packages: {failed_required}")
        return 1
    
    if not failed_optional:
        print("✅ All optional packages installed successfully!")
    else:
        print(f"⚠️  Some optional packages failed: {failed_optional}")
        print("   You can still use the project with limited functionality.")
    
    print("\nYou can now run the project notebooks and scripts!")
    return 0

if __name__ == "__main__":
    exit(main())
```

## Task 6: Update requirements.txt to be More Flexible

### Replace `requirements.txt` with:

```
# Core Python packages
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
scipy>=1.10.0
statsmodels>=0.14.0
pyyaml>=6.0
tqdm>=4.65.0
jupyter>=1.0.0
ipykernel>=6.0.0

# Optional neuroimaging packages (install manually if needed)
# nibabel>=5.0.0
# nilearn>=0.10.0

# Optional AI/ML packages (install manually if needed)  
# torch-geometric>=2.3.0
# pytorch-lightning>=2.0.0
# transformers>=4.30.0
```

## Task 8: Create Prompts Directory and Store Prompt Files

### Create a `prompts/` directory to organize all Cline prompt files:

1. **Create the prompts directory structure**:
   ```
   CGN/
   ├── prompts/
   │   ├── README.md
   │   ├── 01_initial_setup.md
   │   ├── 02_development_environment.md
   │   ├── 03_fix_errors_and_dependencies.md
   │   └── future_prompts/
   ```

2. **Create `CGN/prompts/README.md`**:
   ```markdown
   # CGN Project Prompts
   
   This directory contains all the prompts used with Cline for the CGN (Connectome Graph Networks) project development.
   
   ## Prompt Files
   
   - `01_initial_setup.md` - Initial project structure creation
   - `02_development_environment.md` - Core development environment and first pipeline
   - `03_fix_errors_and_dependencies.md` - Error fixes and dependency management
   - `future_prompts/` - Directory for upcoming development phase prompts
   
   ## Usage
   
   These prompts are designed to be used sequentially with Cline for rapid AI-assisted development of the CGN research project.
   
   ## Prompt Organization
   
   - Each prompt is numbered for sequential execution
   - Prompts build upon previous phases
   - All prompts maintain the CGN directory structure
   - Prompts are designed for 1-2 month rapid research timeline
   ```

3. **Save the current prompt as `CGN/prompts/03_fix_errors_and_dependencies.md`**:
   - Copy the entire content of this current fix prompt into this file
   - This will serve as documentation of the error fixes applied

4. **Create placeholder for future prompts**:
   - Create `CGN/prompts/future_prompts/README.md` with:
   ```markdown
   # Future Development Prompts
   
   This directory will contain prompts for upcoming development phases:
   
   ## Planned Prompts
   - Phase 4: Baseline Models and Initial Graph Neural Networks
   - Phase 5: Brain Graph Transformer Architecture
   - Phase 6: Training Pipeline and Hyperparameter Optimization
   - Phase 7: Evaluation and Cross-Validation
   - Phase 8: Results Analysis and Publication Materials
   ```

## Task 7: Fix Directory Organization

**IMPORTANT**: All files must be created INSIDE the existing CGN directory structure that was already created. Do NOT create new directories outside of CGN.

### Verify and Fix Directory Structure:

1. **Check that you're working inside CGN/**:
   - All file paths should start with `CGN/` 
   - Example: `CGN/src/connectome_analysis/data/loaders.py`
   - NOT: `src/connectome_analysis/data/loaders.py` (outside CGN)

2. **If files were created outside CGN, move them**:
   ```bash
   # Move any files that were created outside CGN back into CGN/
   # For example, if you created scripts/ outside, move it:
   mv scripts/ CGN/scripts/
   mv src/ CGN/src/
   # etc.
   ```

3. **Ensure the following CGN internal structure exists**:
   ```
   CGN/
   ├── src/
   │   └── connectome_analysis/
   │       ├── data/
   │       │   ├── loaders.py          ← Fix this file
   │       │   ├── preprocessing.py     ← Fix this file  
   │       │   └── connectome.py        ← Fix this file
   │       └── __init__.py
   ├── scripts/
   │   ├── data/
   │   │   └── download_datasets.py     ← Fix this file
   │   └── utils/
   │       └── install_dependencies.py  ← Create this file
   ├── notebooks/
   │   └── 01_data_exploration/
   │       └── 01_dataset_overview.ipynb ← Should already exist
   └── requirements.txt                  ← Update this file
   ```

4. **Verify the correct working directory**:
   - When running scripts, you should be IN the CGN directory
   - Paths in scripts should be relative to CGN root
   - Example: `python scripts/utils/install_dependencies.py` (run from inside CGN/)

## Instructions for Cline

1. **FIRST: Verify you're working inside CGN directory structure**
2. **Move any misplaced files back into CGN/ if needed**
3. **Replace all the files** mentioned above with the corrected versions (INSIDE CGN structure)
4. **Create the new installation script** in `CGN/scripts/utils/install_dependencies.py`
5. **Update requirements.txt** inside CGN directory
6. **Test the fixes** by running (from inside CGN directory):
   ```bash
   # Make sure you're in CGN/ directory first
   pwd  # Should show path ending in /CGN
   python scripts/utils/install_dependencies.py
   python -c "import src.connectome_analysis; print('✅ Package imports successfully')"
   ```

## What These Fixes Do

1. **Import Handling**: Added graceful handling for missing packages with informative error messages
2. **Method Signatures**: Fixed inheritance issues in the loader classes
3. **Type Annotations**: Removed problematic type hints that were causing issues
4. **NetworkX Updates**: Updated community detection to use modern NetworkX API
5. **Flexible Installation**: Created an installation script that handles optional dependencies

After these fixes, the project should:
- Import without errors
- Work even if neuroimaging packages aren't installed
- Provide clear error messages about missing optional dependencies
- Be ready for development and testing

The project will now be much more robust and ready for the next development phase!