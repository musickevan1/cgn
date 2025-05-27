import os
import argparse
import pickle
import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image, load as nifti1_load # Import Nifti1Image and load from nifti1
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # Import StratifiedKFold
from typing import Dict, Any, List, Tuple, Optional, cast # Import typing elements and cast
import torch # Import torch
from torch_geometric.data import Data # Import Data object
import nilearn.datasets # Import nilearn.datasets
from nilearn import image # Import nilearn.image

from src.connectome_analysis.data.loaders import ABIDELoader
from src.connectome_analysis.data.connectome import ConnectomeBuilder
from src.connectome_analysis.data.preprocessing import NeuroPreprocessor

def load_and_process_abide(data_dir, atlas_path, atlas_coords_path):
    """Load ABIDE data and create connectome matrices"""
    loader = ABIDELoader(data_dir)
    phenotypic_data = loader.load_phenotypic_data() # Load phenotypic data instead of subjects
    connectome_builder = ConnectomeBuilder() # Initialize with default connectivity_kinds
    preprocessor = NeuroPreprocessor() # Initialize the preprocessor

    connectomes = []
    labels = []
    sites = []
    subject_ids = []

    # Iterate through the rows of the phenotypic data DataFrame
    for index, subject_info in phenotypic_data.iterrows():
        try:
            # Assuming column names in phenotypic data are 'FILE_ID', 'DX_GROUP', 'SITE_ID', 'SUB_ID'
            # Adjust these names if necessary based on the actual ABIDE phenotypic file
            file_id = subject_info['FILE_ID'] # This should be the path to the preprocessed fMRI file
            label = subject_info['DX_GROUP'] # Diagnosis group (e.g., 1 for Autism, 2 for Control)
            
            # Remap labels: 1 -> 0 (ASD), 2 -> 1 (Control) for binary classification
            if label == 1:
                label = 0 # ASD
            elif label == 2:
                label = 1 # Control
            else:
                print(f"Warning: Unexpected DX_GROUP label '{label}' for subject {subject_info['SUB_ID']}. Skipping.")
                continue # Skip subjects with unexpected labels

            site = subject_info['SITE_ID'] # Site ID
            subject_id = subject_info['SUB_ID'] # Subject ID

            # Construct the full file path to the preprocessed fMRI data
            # This assumes a specific directory structure within the data_dir
            # You might need to adjust this path construction based on how the data is organized after download
            # Construct the full file path to the preprocessed fMRI data
            # The file_id from phenotypic data is typically just the subject ID (e.g., 'Pitt_0050003')
            # The actual file name includes '_func_preproc.nii.gz' suffix
            fmri_file_path = os.path.join(data_dir, "abide", "neuroimaging", f"{file_id}_func_preproc.nii.gz")

            if not os.path.exists(fmri_file_path):
                print(f"Warning: fMRI file not found for subject {subject_id} at {fmri_file_path}. Skipping.")
                continue

            # Load the atlas image
            # This assumes atlas_path is a path to a nifti atlas file
            try:
                atlas_img = nifti1_load(atlas_path) # Use nifti1_load
                # Ensure atlas image is 3D, as NiftiLabelsMasker expects 3D atlas
                if atlas_img.ndim == 4:
                    print(f"Warning: Atlas image is 4D (shape: {atlas_img.shape}). Taking first volume.")
                    atlas_img = image.index_img(atlas_img, 0) # Use nilearn.image.index_img to get 3D volume
            except ImportError:
                print("Error: nibabel not installed. Cannot load atlas.")
                continue
            except Exception as e:
                print(f"Error loading atlas file {atlas_path}: {e}. Skipping subject {subject_id}.")
                continue

            # Extract ROI time series using NeuroPreprocessor
            try:
                # Assuming fmri_file_path points to a nifti fMRI image
                fmri_img = nifti1_load(fmri_file_path) # Use nifti1_load
                
                print(f"Debug: fmri_img shape: {cast(Nifti1Image, fmri_img).shape}, ndim: {cast(Nifti1Image, fmri_img).ndim}")
                print(f"Debug: atlas_img shape: {cast(Nifti1Image, atlas_img).shape}, ndim: {cast(Nifti1Image, atlas_img).ndim}")

                roi_time_series = preprocessor.extract_roi_time_series(fmri_img, atlas_img)
            except Exception as e:
                print(f"Error extracting ROI time series for subject {subject_id}: {e}. Skipping.")
                continue

            # Build functional connectome from ROI time series
            # Assuming 'correlation' is the desired connectivity kind for now
            functional_connectomes = connectome_builder.build_functional_connectome(roi_time_series, subject_id)

            # Assuming we take the 'correlation' connectome if available
            if 'correlation' in functional_connectomes:
                processed_connectome = functional_connectomes['correlation']
                connectomes.append(processed_connectome)
                labels.append(label)
                sites.append(site)
                subject_ids.append(subject_id)
            else:
                print(f"Warning: Could not build correlation connectome for subject {subject_id}. Skipping.")
                continue

        except KeyError as e:
            print(f"Error accessing expected column in phenotypic data: {e}. Skipping row {index}.")
            continue
        except Exception as e:
            print(f"Error processing subject with info {subject_info.to_dict()}: {e}")
            continue

    return connectomes, labels, sites, subject_ids

def create_baseline_features(connectomes: List[np.ndarray]) -> np.ndarray:
    """
    Convert connectivity matrices to feature vectors for sklearn models
    - Extract upper triangular matrix (avoid redundancy)
    - Apply Fisher z-transformation to correlations
    - Handle missing values and infinite correlations (basic handling)
    - Feature scaling and selection options (Placeholder)

    Args:
        connectomes: A list of connectivity matrices (numpy arrays).

    Returns:
        A numpy array of flattened and transformed features.
    """
    features = []
    for conn in connectomes:
        # Get upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(conn, k=1)
        flat_features = conn[upper_triangle_indices]

        # Apply Fisher z-transformation
        # Ensure values are within (-1, 1) for arctanh
        flat_features = np.clip(flat_features, -0.9999, 0.9999)
        fisher_z_features = np.arctanh(flat_features)

        # Handle potential NaNs or Infs (e.g., from original data or edge cases)
        # Replace NaNs with 0, Infs with a large number
        fisher_z_features = np.nan_to_num(fisher_z_features, nan=0.0, posinf=1e10, neginf=-1e10)


        features.append(fisher_z_features)

    # Stack features into a single numpy array
    if features:
        return np.vstack(features)
    else:
        return np.array([]) # Return empty array if no connectomes


def create_graph_data(connectomes: List[np.ndarray], labels: List[int], atlas_coords: Optional[np.ndarray] = None, threshold: Optional[float] = None) -> List[Data]:
    """
    Convert connectivity matrices to PyTorch Geometric Data objects
    - Create edge_index and edge_attr from connectivity matrices
    - Handle thresholding for sparse graphs
    - Add node features if available (region coordinates, volumes)
    - Batch multiple subjects into DataLoader format (Handled by DataLoader)

    Args:
        connectomes: A list of connectivity matrices (numpy arrays).
        labels: A list of labels corresponding to the connectomes.
        atlas_coords: Optional numpy array of shape (num_regions, num_dimensions) for node features.
        threshold: Optional threshold for creating sparse graphs (edges where abs(weight) > threshold).

    Returns:
        A list of PyTorch Geometric Data objects.
    """
    graph_data_list: List[Data] = []
    num_regions = connectomes[0].shape[0] if connectomes else 0

    if num_regions == 0:
        print("No connectomes provided to create graph data.")
        return []

    # Create default node features if atlas_coords are not provided
    # Example: Identity matrix or constant features
    if atlas_coords is None:
        print("Atlas coordinates not provided. Using identity matrix as dummy node features.")
        # Identity matrix as node features (num_regions x num_regions)
        default_node_features = torch.eye(num_regions)
    else:
        # Use atlas coordinates as node features
        # Assuming atlas_coords shape is (num_regions, num_dimensions)
        default_node_features = torch.tensor(atlas_coords, dtype=torch.float)


    for i, conn in enumerate(connectomes):
        # Ensure connectivity matrix is a numpy array
        conn = np.asarray(conn)

        # Create edge_index and edge_attr
        if threshold is None:
            # Fully connected graph
            edge_index = torch.combinations(torch.arange(num_regions), 2).t().contiguous()
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Get edge attributes (connectivity values)
            # Need to map the flattened upper triangle values back to edges
            # A simpler way for fully connected: get all pairs (i, j) and their values
            rows, cols = np.triu_indices_from(conn, k=1)
            edge_values = conn[rows, cols]

            # Duplicate values for reverse edges
            edge_attr = torch.tensor(np.concatenate([edge_values, edge_values]), dtype=torch.float).unsqueeze(1) # Shape (num_edges, 1)

            # For fully connected, need to handle self-loops if desired (currently excluded by k=1)
            # If including diagonal (self-loops), use np.triu_indices_from(conn, k=0) and adjust edge_index/attr.
            # Let's stick to no self-loops for now.

        else:
            # Thresholded graph
            # Find indices where absolute connectivity value is above threshold
            rows, cols = np.where(np.abs(conn) > threshold)
            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
            edge_attr = torch.tensor(conn[rows, cols], dtype=torch.float).unsqueeze(1) # Shape (num_edges, 1)


        # Get node features (use default if atlas_coords not provided)
        x = default_node_features.clone() # Clone to avoid modifying the original tensor

        # Get label
        y = torch.tensor([labels[i]], dtype=torch.long) # Assuming labels are long for classification

        # Create PyG Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph_data_list.append(graph_data)

    return graph_data_list


def stratified_split_by_site(data: Any, labels: np.ndarray, sites: np.ndarray, stratify_by_site: bool, n_splits: int, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Create train/val/test split indices respecting site distribution.
    Returns a list of (train_indices, test_indices) tuples for cross-validation.

    Args:
        data: The input data (can be numpy array or list of graph objects). Used for splitting indices.
        labels: Ground truth labels (numpy array). Used for stratification.
        sites: Site labels for each subject (numpy array). Used for site stratification.
        stratify_by_site: Whether to perform Leave-One-Site-Out stratification.
        n_splits: Number of splits for standard K-Fold if not stratifying by site.
        test_size: Proportion of data for the test set in the initial split (not used for LOO-Site).
        val_size: Proportion of data for the validation set (from the remaining train_val data) (not used for LOO-Site).
        random_state: Random state for reproducibility.

    Returns:
        A list of tuples, where each tuple contains (train_indices, test_indices).
        For standard K-Fold, this list has `n_splits` tuples.
        For LOO-Site, this list has `num_unique_sites` tuples.
    """
    n_subjects = len(labels)
    indices = np.arange(n_subjects)
    splits: List[Tuple[List[int], List[int]]] = []

    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. Cannot perform stratified splitting.")
        # Return a single split with all data as train and empty test if only one class
        return [(indices.tolist(), [])]


    if len(np.unique(sites)) < 2 or not stratify_by_site: # Check if site stratification is requested and possible
        # Standard Stratified K-Fold on labels (without site stratification)
        print(f"Creating Standard Stratified K-Fold CV splits with {n_splits} splits.")
        # Use labels for stratification
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # skf.split expects X and y. We can pass indices as X if stratification is only on y.
        for train_indices_np, test_indices_np in skf.split(indices, labels):
             splits.append((train_indices_np.tolist(), test_indices_np.tolist())) # Convert numpy indices to list

    else:
        # Implement Leave-One-Site-Out (LOO) Cross-Validation
        unique_sites = np.unique(sites)
        print(f"Creating Leave-One-Site-Out CV splits across {len(unique_sites)} sites.")

        for test_site in unique_sites:
            train_indices = [i for i, site in enumerate(sites) if site != test_site]
            test_indices = [i for i, site in enumerate(sites) if site == test_site]

            if not train_indices or not test_indices:
                print(f"Warning: Skipping site {test_site} due to insufficient data for splitting.")
                continue

            # For LOO-Site, each split is defined by leaving one site out for testing.
            # The 'train_val_split' logic from the prompt description seems more for a single train/val/test split,
            # not for generating CV folds. Let's stick to generating CV fold indices here.
            # The CrossValidator will handle creating train/test datasets from these indices.

            splits.append((train_indices, test_indices))

    # Note: The prompt's description of stratified_split_by_site also mentions
    # creating train/val/test splits. The current implementation focuses on
    # generating CV fold indices (train/test for each fold).
    # If a single train/val/test split is needed outside of CV, a separate function
    # or logic would be required. For the CrossValidator, a list of train/test indices is standard.
    # Let's assume this function is primarily for generating CV indices.

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ABIDE data for CGN project.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw ABIDE data.")
    parser.add_argument("--atlas_path", type=str, help="Path to the brain atlas file (optional). If not provided, MSDL atlas will be fetched.")
    parser.add_argument("--atlas_coords_path", type=str, help="Path to the brain atlas coordinates file (optional). If not provided, MSDL atlas coordinates will be fetched.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for the test set.")
    parser.add_argument("--val_size", type=float, default=0.25, help="Proportion of data for the validation set (from train_val).")
    parser.add_argument("--graph_threshold", type=float, help="Threshold for creating sparse graph data.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds for standard K-Fold CV.")
    parser.add_argument("--stratify_by_site", action='store_true', help="Use Leave-One-Site-Out stratification.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility of splits.")


    args = parser.parse_args()

    # Handle atlas fetching if paths are not provided
    current_atlas_path = args.atlas_path
    current_atlas_coords_path = args.atlas_coords_path

    if not current_atlas_path or not current_atlas_coords_path:
        print("Atlas paths not provided. Fetching MSDL atlas from Nilearn...")
        try:
            msdl_atlas = nilearn.datasets.fetch_atlas_msdl()
            print(f"Type of msdl_atlas: {type(msdl_atlas)}")
            if hasattr(msdl_atlas, 'keys'):
                print(f"Keys in msdl_atlas: {msdl_atlas.keys()}")
            
            current_atlas_path = msdl_atlas['maps']
            fetched_atlas_coords = np.array(msdl_atlas['region_coords']) # Convert to numpy array here
            print(f"Fetched MSDL atlas: {current_atlas_path}")
            print(f"Fetched MSDL atlas coordinates: {fetched_atlas_coords}")
        except Exception as e:
            print(f"Error fetching MSDL atlas: {e}. Please ensure nilearn is installed and you have an internet connection.")
            print("Exiting as atlas is required for connectome building.")
            exit()
    else:
        fetched_atlas_coords = None # No atlas fetched if paths were provided

    print("Loading and processing ABIDE data...")
    # Assuming load_and_process_abide returns connectivity matrices (numpy arrays)
    connectomes, labels, sites, subject_ids = load_and_process_abide(args.data_dir, current_atlas_path, current_atlas_coords_path) # Pass original args for load_and_process_abide
    print(f"Loaded and processed {len(connectomes)} subjects.")

    if not connectomes:
        print("No subjects processed successfully. Exiting.")
        exit()

    # Convert labels and sites to numpy arrays for consistent splitting
    labels_np = np.array(labels)
    sites_np = np.array(sites)


    print("Creating baseline features...")
    baseline_features = create_baseline_features(connectomes)
    print(f"Created baseline features with shape: {baseline_features.shape}")

    print("Creating graph data...")
    # Use fetched_atlas_coords if available, otherwise try to load from current_atlas_coords_path
    atlas_coords_for_graph = None
    if fetched_atlas_coords is not None:
        atlas_coords_for_graph = fetched_atlas_coords
        print(f"Using fetched atlas coordinates for graph data with shape: {atlas_coords_for_graph.shape}")
    elif current_atlas_coords_path and os.path.exists(current_atlas_coords_path):
        try:
            atlas_coords_for_graph = np.load(current_atlas_coords_path)
            print(f"Loaded atlas coordinates from path for graph data with shape: {atlas_coords_for_graph.shape}")
        except Exception as e:
            print(f"Warning: Could not load atlas coordinates from {current_atlas_coords_path}: {e}. Proceeding without node features.")
            atlas_coords_for_graph = None
    else:
        print("Warning: Atlas coordinates not available for graph data. Proceeding without node features.")
        atlas_coords_for_graph = None

    graph_data_list = create_graph_data(connectomes, labels, atlas_coords=atlas_coords_for_graph, threshold=args.graph_threshold)
    print(f"Created {len(graph_data_list)} graph data objects.")

    print("Creating stratified train/val/test splits indices...")
    # The stratified_split_by_site function now returns indices for CV folds, not a single split.
    # The prompt's main block seems to expect a single train/val/test split for saving.
    # Let's adapt the main block to create a single split for saving purposes,
    # while the function itself is designed for CV indices.
    # A separate function for single train/val/test split might be cleaner,
    # but for completing the placeholder script, let's adapt the main block.

    # Create a single train/val/test split for saving
    n_subjects = len(labels_np)
    indices = np.arange(n_subjects)

    # First split: train_val and test
    # Use the stratify_by_site argument from command line
    if args.stratify_by_site:
         # For LOO-Site, the concept of a single train/val/test split is less standard.
         # The LOO-Site splits are the CV folds themselves.
         # If we need a single representative split for saving, we could take the first LOO split,
         # or a random stratified split. Let's create a single stratified split by labels for saving,
         # as site stratification is primarily for the CV loop.
         print("Creating a single stratified split by labels for saving (site stratification is for CV).")
         sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
         train_val_indices_np, test_indices_np = next(sss_test.split(indices, labels_np)) # Stratify by labels

         # Second split: train and val from train_val
         adjusted_val_size = args.val_size / (1 - args.test_size)
         sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=args.random_state)
         train_indices_np, val_indices_np = next(sss_val.split(train_val_indices_np, labels_np[train_val_indices_np])) # Stratify by labels

    else:
        # Standard stratified split by labels
        print("Creating a single stratified split by labels for saving.")
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_val_indices_np, test_indices_np = next(sss_test.split(indices, labels_np)) # Stratify by labels

        # Second split: train and val from train_val
        adjusted_val_size = args.val_size / (1 - args.test_size)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=args.random_state)
        train_indices_np, val_indices_np = next(sss_val.split(train_val_indices_np, labels_np[train_val_indices_np])) # Stratify by labels


    # Convert numpy indices to lists
    train_indices = train_indices_np.tolist()
    val_indices = val_indices_np.tolist()
    test_indices = test_indices_np.tolist()


    # Extract data for each split using the indices
    # Need to handle both baseline features and graph data
    train_baseline_features = baseline_features[train_indices] if baseline_features.size > 0 else np.array([])
    val_baseline_features = baseline_features[val_indices] if baseline_features.size > 0 else np.array([])
    test_baseline_features = baseline_features[test_indices] if baseline_features.size > 0 else np.array([])

    # For graph data, select the Data objects by index
    train_graph_data = [graph_data_list[i] for i in train_indices]
    val_graph_data = [graph_data_list[i] for i in val_indices]
    test_graph_data = [graph_data_list[i] for i in test_indices]

    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]

    train_sites = [sites[i] for i in train_indices]
    val_sites = [sites[i] for i in val_indices]
    test_sites = [sites[i] for i in test_indices]


    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")

    # Save processed data and splits
    os.makedirs(args.output_dir, exist_ok=True)

    # Save full processed data (optional, might be large)
    # with open(os.path.join(args.output_dir, "connectomes.pkl"), "wb") as f:
    #     pickle.dump(connectomes, f)
    with open(os.path.join(args.output_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    with open(os.path.join(args.output_dir, "sites.pkl"), "wb") as f:
        pickle.dump(sites, f)
    with open(os.path.join(args.output_dir, "subject_ids.pkl"), "wb") as f:
        pickle.dump(subject_ids, f)


    # Save baseline features and graph data lists
    with open(os.path.join(args.output_dir, "baseline_features.pkl"), "wb") as f:
        pickle.dump(baseline_features, f)
    with open(os.path.join(args.output_dir, "graph_data_list.pkl"), "wb") as f:
        pickle.dump(graph_data_list, f)


    # Save split indices
    with open(os.path.join(args.output_dir, "train_indices.pkl"), "wb") as f:
        pickle.dump(train_indices, f)
    with open(os.path.join(args.output_dir, "val_indices.pkl"), "wb") as f:
        pickle.dump(val_indices, f)
    with open(os.path.join(args.output_dir, "test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)

    # Save split data (optional, might be redundant if saving full data and indices)
    # This might be useful if the dataset loading/processing is slow and splits are fixed.
    # However, for flexibility with CrossValidator, saving indices is usually sufficient.
    # Let's save the split data as requested by the original placeholder structure.
    with open(os.path.join(args.output_dir, "train_split_data.pkl"), "wb") as f:
        # Save both baseline and graph data for the split
        pickle.dump({"baseline_features": train_baseline_features, "graph_data": train_graph_data, "labels": train_labels, "sites": train_sites}, f)
    with open(os.path.join(args.output_dir, "val_split_data.pkl"), "wb") as f:
        pickle.dump({"baseline_features": val_baseline_features, "graph_data": val_graph_data, "labels": val_labels, "sites": val_sites}, f)
    with open(os.path.join(args.output_dir, "test_split_data.pkl"), "wb") as f:
        pickle.dump({"baseline_features": test_baseline_features, "graph_data": test_graph_data, "labels": test_labels, "sites": test_sites}, f)


    print(f"Processed data, graph data, and split indices/data saved to {args.output_dir}")
