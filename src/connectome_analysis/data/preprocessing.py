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

# Import handling for optional neuroimaging packages with explicit None assignment
try:
    import nibabel as nib
    from nilearn import image, masking, signal
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    nib = None  # Explicit None assignment
    image = None  # Explicit None assignment  
    masking = None  # Explicit None assignment
    signal = None  # Explicit None assignment
    ConnectivityMeasure = None  # Explicit None assignment
    NiftiLabelsMasker = None  # Explicit None assignment
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
        """Initialize preprocessor with common parameters."""
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        
    def preprocess_fmri(self, fmri_img: Any, mask_img: Any = None) -> Any:
        """Preprocess fMRI data with standard pipeline."""
        
        if not NEUROIMAGING_AVAILABLE or masking is None or signal is None:
            raise ImportError("Neuroimaging packages required for preprocessing")
        
        # Apply brain mask if provided
        if mask_img is not None:
            time_series = masking.apply_mask(fmri_img, mask_img)
        else:
            # Create mask from data
            mask_img = masking.compute_epi_mask(fmri_img)
            time_series = masking.apply_mask(fmri_img, mask_img)
        
        # Apply temporal filtering and standardization
        # Note: signal.clean expects standardize as boolean, 'zscore', or 'psc'
        time_series = signal.clean(
            time_series,
            detrend=self.detrend,
            standardize="zscore",
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r
        )
        
        return np.array(time_series)
    
    def extract_roi_time_series(self, fmri_img: Any, atlas_img: Any) -> Any:
        """Extract ROI time series using an atlas."""
        
        if not NEUROIMAGING_AVAILABLE or NiftiLabelsMasker is None:
            raise ImportError("Neuroimaging packages required for ROI extraction")
        
        # Debugging: Check dimensionality of inputs
        # Accessing .shape and .ndim directly on Niimg-like objects is common in nilearn context
        # but Pylance might not recognize it without explicit type hinting or casting.
        # We'll proceed with these checks for runtime debugging.
        if hasattr(fmri_img, 'shape') and fmri_img.ndim != 4:
            warnings.warn(f"fMRI image is not 4D (shape: {fmri_img.shape}, ndim: {fmri_img.ndim}). NiftiLabelsMasker expects 4D for time series extraction.")
        if hasattr(atlas_img, 'shape') and atlas_img.ndim != 3:
            warnings.warn(f"Atlas image is not 3D (shape: {atlas_img.shape}, ndim: {atlas_img.ndim}). NiftiLabelsMasker expects 3D atlas.")

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
        return np.array(roi_time_series)

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
            fd_values = confounds['framewise_displacement'].values
        else:
            # Approximate FD from motion parameters
            motion_params = confounds[motion_columns].values
            fd_values = np.sum(np.abs(np.diff(motion_params, axis=0)), axis=1)
        
        # Convert to numpy arrays and handle types properly
        fd_array = np.array(fd_values, dtype=float)
        fd_clean = fd_array[~np.isnan(fd_array)]
        
        return {
            'mean_fd': float(np.mean(fd_clean)) if len(fd_clean) > 0 else np.nan,
            'max_fd': float(np.max(fd_clean)) if len(fd_clean) > 0 else np.nan,
            'n_high_motion': int(np.sum(fd_clean > 0.5)) if len(fd_clean) > 0 else 0
        }
    
    @staticmethod
    def compute_signal_metrics(time_series: np.ndarray) -> Dict[str, float]:
        """Compute signal quality metrics."""
        # Temporal signal-to-noise ratio
        mean_signal = np.mean(time_series, axis=0)
        std_signal = np.std(time_series, axis=0)
        
        # Avoid division by zero
        std_signal[std_signal == 0] = np.finfo(float).eps
        tsnr = mean_signal / std_signal
        
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
