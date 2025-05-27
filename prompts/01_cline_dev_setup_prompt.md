# Cline Development Setup - Phase 2: Environment & First Pipeline

## Current Status
The CGN directory structure has been created successfully. Now we need to set up the development environment and create the first functional data processing pipeline.

## Task Overview
1. Set up Python package structure with core modules
2. Create data downloading and preprocessing utilities
3. Build the first notebook for data exploration
4. Implement basic connectome construction pipeline
5. Set up development environment configuration

## Step 1: Core Python Package Implementation

### Create `src/connectome_analysis/data/loaders.py`
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
import requests
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets
import warnings

class DatasetLoader:
    """Base class for neuroimaging dataset loading."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name: str) -> Dict:
        """Download dataset using nilearn's built-in downloaders."""
        raise NotImplementedError
        
    def load_phenotypic_data(self, dataset_name: str) -> pd.DataFrame:
        """Load phenotypic/demographic data."""
        raise NotImplementedError
        
    def get_subject_list(self, dataset_name: str) -> List[str]:
        """Get list of available subjects."""
        raise NotImplementedError

class ABIDELoader(DatasetLoader):
    """Loader for ABIDE (Autism Brain Imaging Data Exchange) dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        super().__init__(data_dir)
        self.dataset_dir = self.data_dir / "abide"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_dataset(self) -> Dict:
        """Download ABIDE dataset using nilearn."""
        print("Downloading ABIDE dataset...")
        
        # Use nilearn's ABIDE dataset fetcher
        abide_data = datasets.fetch_abide_pcp(
            data_dir=str(self.dataset_dir),
            pipeline='cpac',
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=['func_preproc'],
            quality_checked=True
        )
        
        return {
            'functional_data': abide_data.func_preproc,
            'phenotypic': abide_data.phenotypic,
            'description': abide_data.description
        }
    
    def load_phenotypic_data(self) -> pd.DataFrame:
        """Load ABIDE phenotypic data."""
        # First try to download if not exists
        try:
            abide_data = self.download_dataset()
            return abide_data['phenotypic']
        except Exception as e:
            print(f"Error loading ABIDE data: {e}")
            return pd.DataFrame()

class ADHD200Loader(DatasetLoader):
    """Loader for ADHD-200 dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        super().__init__(data_dir)
        self.dataset_dir = self.data_dir / "adhd200"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_dataset(self) -> Dict:
        """Download ADHD-200 dataset using nilearn."""
        print("Downloading ADHD-200 dataset...")
        
        adhd_data = datasets.fetch_adhd(
            data_dir=str(self.dataset_dir),
            n_subjects=40  # Start with subset for rapid development
        )
        
        return {
            'functional_data': adhd_data.func,
            'phenotypic': adhd_data.phenotypic,
            'confounds': adhd_data.confounds
        }

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

### Create `src/connectome_analysis/data/preprocessing.py`
```python
"""
Preprocessing utilities for neuroimaging data.
Handles basic preprocessing, quality control, and standardization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import nibabel as nib
from nilearn import image, masking, signal
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
import warnings

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
                       fmri_img: nib.Nifti1Image,
                       mask_img: Optional[nib.Nifti1Image] = None) -> np.ndarray:
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
                               fmri_img: nib.Nifti1Image,
                               atlas_img: nib.Nifti1Image) -> np.ndarray:
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
        
        from nilearn.input_data import NiftiLabelsMasker
        
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
        
        return {
            'mean_fd': np.nanmean(fd),
            'max_fd': np.nanmax(fd),
            'n_high_motion': np.sum(fd > 0.5)  # Frames with FD > 0.5mm
        }
    
    @staticmethod
    def compute_signal_metrics(time_series: np.ndarray) -> Dict[str, float]:
        """Compute signal quality metrics."""
        # Temporal signal-to-noise ratio
        tsnr = np.mean(time_series, axis=0) / np.std(time_series, axis=0)
        
        # Global signal properties
        global_signal = np.mean(time_series, axis=1)
        
        return {
            'mean_tsnr': np.nanmean(tsnr),
            'global_signal_std': np.std(global_signal),
            'n_outlier_volumes': np.sum(np.abs(global_signal - np.mean(global_signal)) > 3 * np.std(global_signal))
        }

def load_standard_atlases() -> Dict[str, str]:
    """Download and return paths to standard brain atlases."""
    from nilearn import datasets
    
    atlases = {}
    
    # AAL atlas
    try:
        aal = datasets.fetch_atlas_aal(version='SPM12')
        atlases['aal'] = aal.maps
    except:
        print("Could not download AAL atlas")
    
    # Schaefer atlas
    try:
        schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        atlases['schaefer_400'] = schaefer.maps
    except:
        print("Could not download Schaefer atlas")
    
    return atlases
```

### Create `src/connectome_analysis/data/connectome.py`
```python
"""
Connectome construction utilities.
Build functional and structural connectivity matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
import warnings

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
        for kind in connectivity_kinds:
            self.connectivity_measures[kind] = ConnectivityMeasure(
                kind=kind,
                standardize=standardize_connectomes
            )
    
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
        
        for kind, measure in self.connectivity_measures.items():
            try:
                # Fit and transform time series
                connectivity_matrix = measure.fit_transform([roi_time_series])[0]
                connectomes[kind] = connectivity_matrix
                
            except Exception as e:
                print(f"Warning: Could not compute {kind} connectivity for {subject_id}: {e}")
                connectomes[kind] = np.full((roi_time_series.shape[1], roi_time_series.shape[1]), np.nan)
        
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
            
            # Modularity
            try:
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
            except:
                modularity = np.nan
            
            # Small-worldness
            if not np.isnan(clustering) and not np.isnan(path_length):
                # Generate random graph for comparison
                n_nodes = len(G.nodes())
                n_edges = len(G.edges())
                random_G = nx.erdos_renyi_graph(n_nodes, n_edges / (n_nodes * (n_nodes - 1) / 2))
                
                random_clustering = nx.average_clustering(random_G)
                random_path_length = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else np.nan
                
                if not np.isnan(random_clustering) and not np.isnan(random_path_length):
                    small_worldness = (clustering / random_clustering) / (path_length / random_path_length)
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

## Step 2: Create First Data Exploration Notebook

### Create `notebooks/01_data_exploration/01_dataset_overview.ipynb`
Create a Jupyter notebook with the following structure:

```python
# Cell 1: Setup and imports
import sys
sys.path.append('../../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from connectome_analysis.data.loaders import create_dataset_loader, get_available_datasets
from connectome_analysis.data.preprocessing import NeuroPreprocessor, load_standard_atlases

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("CGN: Connectome Graph Networks - Data Exploration")
print("=" * 50)

# Cell 2: Check available datasets
print("Available datasets:")
datasets = get_available_datasets()
for dataset in datasets:
    print(f"  - {dataset}")

# Cell 3: Load ABIDE dataset
print("\nLoading ABIDE dataset...")
abide_loader = create_dataset_loader('abide', data_dir='../../data/raw')

# Try to load phenotypic data
try:
    phenotypic_data = abide_loader.load_phenotypic_data()
    print(f"Loaded phenotypic data: {phenotypic_data.shape[0]} subjects")
    print(f"Columns: {list(phenotypic_data.columns)}")
except Exception as e:
    print(f"Could not load ABIDE data: {e}")
    print("This is normal for first run - data will be downloaded when needed")

# Cell 4: Explore data structure
if 'phenotypic_data' in locals() and not phenotypic_data.empty:
    print("\nDataset Overview:")
    print(f"Total subjects: {len(phenotypic_data)}")
    
    if 'DX_GROUP' in phenotypic_data.columns:
        print("\nDiagnosis distribution:")
        print(phenotypic_data['DX_GROUP'].value_counts())
    
    if 'AGE_AT_SCAN' in phenotypic_data.columns:
        print(f"\nAge range: {phenotypic_data['AGE_AT_SCAN'].min():.1f} - {phenotypic_data['AGE_AT_SCAN'].max():.1f}")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(phenotypic_data['AGE_AT_SCAN'], bins=20, alpha=0.7)
        plt.xlabel('Age at Scan')
        plt.ylabel('Count')
        plt.title('Age Distribution')
        
        if 'DX_GROUP' in phenotypic_data.columns:
            plt.subplot(1, 2, 2)
            for group in phenotypic_data['DX_GROUP'].unique():
                if not pd.isna(group):
                    subset = phenotypic_data[phenotypic_data['DX_GROUP'] == group]
                    plt.hist(subset['AGE_AT_SCAN'], bins=15, alpha=0.7, label=f'Group {group}')
            plt.xlabel('Age at Scan')
            plt.ylabel('Count')
            plt.title('Age Distribution by Group')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# Cell 5: Check available atlases
print("\nChecking available brain atlases...")
try:
    atlases = load_standard_atlases()
    print("Available atlases:")
    for name, path in atlases.items():
        print(f"  - {name}: {path}")
except Exception as e:
    print(f"Could not load atlases: {e}")

# Cell 6: Summary and next steps
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("✅ Project structure created")
print("✅ Data loading utilities implemented")
print("✅ Basic preprocessing pipeline ready")
print("✅ Atlas integration available")
print("\nNext steps:")
print("1. Download and preprocess neuroimaging data")
print("2. Build first connectome matrices")
print("3. Implement baseline classification models")
print("4. Develop graph transformer architecture")
```

## Step 3: Create Data Download Script

### Create `scripts/data/download_datasets.py`
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
    loader = create_dataset_loader(args.dataset, args.data_dir)
    
    # Download dataset
    try:
        data = loader.download_dataset()
        print(f"✅ Successfully downloaded {args.dataset} dataset")
        
        if 'phenotypic' in data:
            print(f"📊 Phenotypic data shape: {data['phenotypic'].shape}")
        if 'functional_data' in data:
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

## Step 4: Update Main Package Init Files

### Update `src/connectome_analysis/__init__.py`
```python
"""
CGN: Connectome Graph Networks
AI-Augmented Connectome Analysis using Graph Neural Networks

This package provides tools for:
- Loading and preprocessing neuroimaging datasets
- Constructing functional and structural connectomes  
- Building graph neural network models for brain analysis
- Evaluating model performance and interpretability
"""

__version__ = "0.1.0"
__author__ = "CGN Research Team"

# Import main modules
from . import data
from . import models
from . import training
from . import evaluation
from . import visualization
from . import utils

# Main classes for easy access
from .data.loaders import create_dataset_loader, get_available_datasets
from .data.preprocessing import NeuroPreprocessor, QualityControl
from .data.connectome import ConnectomeBuilder, MultiModalConnectome
```

### Update `src/connectome_analysis/data/__init__.py`
```python
"""
Data processing module for CGN project.
Handles loading, preprocessing, and connectome construction.
"""

from .loaders import create_dataset_loader, get_available_datasets, ABIDELoader, ADHD200Loader
from .preprocessing import NeuroPreprocessor, QualityControl, load_standard_atlases
from .connectome import ConnectomeBuilder, MultiModalConnectome, extract_connectome_features

__all__ = [
    'create_dataset_loader', 'get_available_datasets', 'ABIDELoader', 'ADHD200Loader',
    'NeuroPreprocessor', 'QualityControl', 'load_standard_atlases',
    'ConnectomeBuilder', 'MultiModalConnectome', 'extract_connectome_features'
]
```

## Instructions for Cline

1. **Create all the Python files** listed above with the exact content provided
2. **Create the Jupyter notebook** with the cell structure shown
3. **Create the download script** and make it executable
4. **Update the __init__.py files** as specified
5. **Test the setup** by running:
   ```bash
   cd CGN
   python -c "import src.connectome_analysis; print('✅ Package imports successfully')"
   ```

## Priority Order
1. Create the core Python modules (loaders.py, preprocessing.py, connectome.py)
2. Update the __init__.py files
3. Create the exploration notebook
4. Create the download script
5. Test imports and basic functionality

This creates a functional foundation where you can immediately start exploring data and building connectomes. The code is designed to work with free/open datasets and uses only open-source libraries.

## Next Phase Preview
After this is complete, we'll move to:
1. Running the first notebook to explore data
2. Building baseline models (traditional ML)
3. Implementing the graph transformer architecture
4. Setting up training and evaluation pipelines