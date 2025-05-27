import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import nibabel as nib
from nibabel.nifti1 import Nifti1Image # Correct import for Nifti1Image
from nilearn import plotting, regions
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_schaefer_2018, load_mni152_template # Correct import for load_mni152_template

def load_atlas_info(atlas_name: str = "schaefer") -> Dict[str, Any]:
    """
    Loads information for a specified brain atlas.

    Args:
        atlas_name (str): Name of the atlas ('aal' or 'schaefer').

    Returns:
        Dict[str, Any]: Dictionary containing atlas data (maps, labels, networks).
    """
    if atlas_name == "aal":
        atlas = fetch_atlas_aal()
        return {
            "maps": atlas.maps,
            "labels": atlas.labels,
            "description": "AAL (Automated Anatomical Labeling) atlas"
        }
    elif atlas_name == "schaefer":
        # Schaefer 2018 atlas with 400 parcels and 7 networks
        atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)
        return {
            "maps": atlas.maps,
            "labels": atlas.labels,
            "networks": atlas.networks,
            "description": "Schaefer 2018 atlas (400 parcels, 7 networks)"
        }
    else:
        raise ValueError(f"Unsupported atlas name: {atlas_name}. Choose 'aal' or 'schaefer'.")

def map_features_to_brain_regions(
    feature_importances: np.ndarray,
    atlas_labels: List[str]
) -> pd.DataFrame:
    """
    Maps feature importance scores to brain regions based on atlas labels.

    Args:
        feature_importances (np.ndarray): 1D array of feature importance scores.
        atlas_labels (List[str]): List of brain region labels from the atlas,
                                  corresponding to the features.

    Returns:
        pd.DataFrame: DataFrame with brain region, importance score, and potentially network info.
    """
    if len(feature_importances) != len(atlas_labels):
        raise ValueError("Length of feature_importances must match length of atlas_labels.")

    df = pd.DataFrame({
        "region": atlas_labels,
        "importance": feature_importances
    })
    # Add network information if available (e.g., for Schaefer atlas)
    # This would require parsing Schaefer labels or having a separate mapping.
    # Example: '7Networks_17A_LH_Vis_1' -> 'Visual Network'
    
    # Placeholder for network mapping
    df['network'] = "Unknown"
    if "7Networks" in atlas_labels[0]: # Heuristic for Schaefer
        df['network'] = df['region'].apply(lambda x: x.split('_')[0] if '_' in x else "Unknown")

    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

def visualize_brain_map(
    importance_df: pd.DataFrame,
    atlas_maps_img: Nifti1Image,
    atlas_labels: List[str],
    title: str = "Brain Map of Feature Importance",
    threshold: Optional[float] = None,
    output_path: Optional[str] = None
):
    """
    Visualizes feature importance on a brain map.

    Args:
        importance_df (pd.DataFrame): DataFrame with 'region' and 'importance' columns.
        atlas_maps_img (nib.Nifti1Image): Nifti image of the atlas maps.
        atlas_labels (List[str]): List of brain region labels from the atlas.
        title (str): Title of the plot.
        threshold (Optional[float]): Minimum importance value to display.
        output_path (Optional[str]): Path to save the plot. If None, displays the plot.
    """
    # Create a dictionary mapping region labels to importance values
    label_to_importance = dict(zip(importance_df['region'], importance_df['importance']))

    # Create a new Nifti image with importance values
    # This requires mapping importance values back to the atlas parcels
    
    # Get the data array from the Nifti image
    atlas_data = atlas_maps_img.get_fdata()
    
    # Create an empty array for the importance map
    importance_map_data = np.zeros_like(atlas_data, dtype=float)
    
    # Iterate through each unique parcel ID in the atlas data
    unique_parcel_ids = np.unique(atlas_data)
    for parcel_id in unique_parcel_ids:
        if parcel_id == 0: # Skip background
            continue
        
        # Find the corresponding label for this parcel_id
        # This assumes atlas_labels are ordered by parcel_id or there's a direct mapping
        # For AAL, labels are typically 1-indexed. For Schaefer, it depends.
        # Need a robust way to map parcel_id to label.
        # For now, assuming parcel_id directly corresponds to index in atlas_labels (minus 1 if 1-indexed)
        
        # Placeholder for robust parcel_id to label mapping
        try:
            # Assuming atlas_labels are 1-indexed for AAL, 0-indexed for Schaefer
            if "aal" in atlas_labels[0].lower(): # Heuristic for AAL
                label_idx = int(parcel_id) - 1
            else: # Assume 0-indexed for Schaefer
                label_idx = int(parcel_id)
            
            region_label = atlas_labels[label_idx]
            importance_value = label_to_importance.get(region_label, 0.0)
            
            # Assign importance value to all voxels belonging to this parcel
            importance_map_data[atlas_data == parcel_id] = importance_value
        except (IndexError, KeyError):
            # print(f"Warning: Could not map parcel ID {parcel_id} to a known region label.")
            pass # Keep importance as 0.0 for unmapped regions

    # Create a new Nifti image from the importance map data
    importance_map_img = Nifti1Image(importance_map_data, atlas_maps_img.affine)

    # Plot the brain map
    # Handle threshold being None for plot_stat_map
    plot_threshold = threshold if threshold is not None else 0.0 # Default to 0.0 if None
    
    display = plotting.plot_stat_map(
        importance_map_img,
        bg_img=load_mni152_template(), # Use MNI template as background
        title=title,
        threshold=plot_threshold,
        cmap='hot_r', # Red for high importance
        cut_coords=None, # Auto-select cuts
        display_mode='ortho' # Orthogonal views
    )
    
    if output_path:
        if display: # Ensure display object is not None
            display.savefig(output_path)
        else:
            print("Warning: Display object is None, cannot save figure.")
    else:
        plotting.show()
