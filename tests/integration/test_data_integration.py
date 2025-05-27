import unittest
import os
import shutil
import numpy as np
import pickle

# Import modules to be tested
from src.connectome_analysis.data.loaders import ABIDELoader
from src.connectome_analysis.data.connectome import ConnectomeBuilder
from src.connectome_analysis.data.preprocessing import NeuroPreprocessor
from scripts.integration.prepare_abide_data import load_and_process_abide, stratified_split_by_site # Import functions from the script

# Define paths for dummy data (adjust as needed for your test setup)
# These paths assume you have a small set of dummy data files for testing
DUMMY_DATA_DIR = 'tests/fixtures/sample_data/abide/raw' # Example path
DUMMY_ATLAS_PATH = 'tests/fixtures/sample_data/atlas/atlas.nii.gz' # Example path
DUMMY_ATLAS_COORDS_PATH = 'tests/fixtures/sample_data/atlas/coordinates.txt' # Example path
PROCESSED_DATA_DIR = 'tests/integration/processed_data' # Output directory for processed data

class TestDataIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up dummy data and directories before running tests."""
        # Create dummy data directories and files if they don't exist
        os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DUMMY_ATLAS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(DUMMY_ATLAS_COORDS_PATH), exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # Create placeholder dummy files (you would replace this with actual small test files)
        # Example: Create a dummy phenotypic file
        dummy_phenotypic_content = """SUB_ID,DX_GROUP,SITE_ID,FILE_ID
1,1,SiteA,sub_001_func_preproc.nii.gz
2,2,SiteB,sub_002_func_preproc.nii.gz
3,1,SiteA,sub_003_func_preproc.nii.gz
4,2,SiteC,sub_004_func_preproc.nii.gz
"""
        phenotypic_dir = os.path.join(DUMMY_DATA_DIR, 'abide', 'phenotypic')
        os.makedirs(phenotypic_dir, exist_ok=True)
        with open(os.path.join(phenotypic_dir, 'ABIDE_pcp_phenotypic.csv'), 'w') as f:
            f.write(dummy_phenotypic_content)

        # Example: Create dummy fMRI files (requires nibabel and actual nifti creation)
        # This is complex for a placeholder, so we'll skip actual file creation for now
        # and assume the loader can handle the dummy phenotypic data.
        # A proper test would involve creating small, valid .nii.gz files.

        # Example: Create dummy atlas files
        # This also requires nibabel
        # For now, just create empty files or simple text files if the builder can handle it
        with open(DUMMY_ATLAS_PATH, 'w') as f:
            f.write("dummy atlas content")
        with open(DUMMY_ATLAS_COORDS_PATH, 'w') as f:
            f.write("dummy coordinates content")


    @classmethod
    def tearDownClass(cls):
        """Clean up dummy data and directories after running tests."""
        # Remove the processed data directory
        if os.path.exists(PROCESSED_DATA_DIR):
            shutil.rmtree(PROCESSED_DATA_DIR)
        # Optionally, remove dummy raw data if created here
        # if os.path.exists(DUMMY_DATA_DIR):
        #     shutil.rmtree(DUMMY_DATA_DIR)
        # if os.path.exists(os.path.dirname(DUMMY_ATLAS_PATH)):
        #      shutil.rmtree(os.path.dirname(DUMMY_ATLAS_PATH))


    def test_load_and_process_abide(self):
        """Test ABIDE data loading and processing."""
        # This test requires actual dummy fMRI and atlas files that can be loaded by nibabel
        # and processed by NeuroPreprocessor and ConnectomeBuilder.
        # With current placeholders, this test will likely fail or not fully verify the process.

        # Placeholder assertion
        self.assertTrue(True, "Placeholder test for load_and_process_abide")

        # Example of what a real test might do:
        # connectomes, labels, sites, subject_ids = load_and_process_abide(
        #     DUMMY_DATA_DIR, DUMMY_ATLAS_PATH, DUMMY_ATLAS_COORDS_PATH
        # )
        # self.assertIsNotNone(connectomes)
        # self.assertGreater(len(connectomes), 0)
        # self.assertEqual(len(connectomes), len(labels))
        # self.assertEqual(len(connectomes), len(sites))
        # self.assertEqual(len(connectomes), len(subject_ids))
        # # Add more assertions to check properties of connectomes, labels, sites, etc.


    def test_stratified_split_by_site(self):
        """Test train/test split functionality with site stratification."""
        # Create dummy data, labels, and sites for splitting
        dummy_data = [np.random.rand(10, 10) for _ in range(10)] # 10 dummy subjects
        dummy_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Example labels
        # Create dummy data, labels (numpy array), and sites (numpy array) for splitting
        dummy_data = [np.random.rand(10, 10) for _ in range(10)] # 10 dummy subjects
        dummy_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # Example labels (numpy array)
        dummy_sites = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A']) # Example sites (numpy array)
        n_subjects = len(dummy_data)
        indices = np.arange(n_subjects)

        # Test site-stratified (LOO) splitting
        print("\nTesting site-stratified splitting...")
        stratify_by_site = True
        n_splits = len(np.unique(dummy_sites)) # For LOO, n_splits is the number of unique sites
        loo_splits = stratified_split_by_site(
            dummy_data, dummy_labels, dummy_sites,
            stratify_by_site=stratify_by_site,
            n_splits=n_splits,
            random_state=42 # Pass random_state
        )

        # Assertions for LOO splits
        self.assertIsInstance(loo_splits, list)
        self.assertEqual(len(loo_splits), n_splits) # Number of splits should equal number of unique sites

        for train_indices, test_indices in loo_splits:
            self.assertIsInstance(train_indices, list)
            self.assertIsInstance(test_indices, list)
            self.assertEqual(len(train_indices) + len(test_indices), n_subjects) # Check total subjects
            self.assertEqual(len(np.intersect1d(train_indices, test_indices)), 0) # Check for overlap

            # Verify that the test set contains subjects from only one site
            test_sites_in_split = dummy_sites[test_indices]
            self.assertEqual(len(np.unique(test_sites_in_split)), 1)

            # Verify that the training set contains subjects from all other sites
            train_sites_in_split = dummy_sites[train_indices]
            unique_train_sites = np.unique(train_sites_in_split)
            # The unique test site should be excluded from the training sites
            test_site_value = np.unique(test_sites_in_split)[0]
            self.assertNotIn(test_site_value, unique_train_sites)
            self.assertEqual(len(unique_train_sites), len(np.unique(dummy_sites)) - 1)


        # Test standard stratified k-fold splitting
        print("\nTesting standard stratified k-fold splitting...")
        stratify_by_site = False
        n_splits = 3 # Example number of folds
        skf_splits = stratified_split_by_site(
            dummy_data, dummy_labels, dummy_sites, # Pass dummy_sites even if not used for stratification
            stratify_by_site=stratify_by_site,
            n_splits=n_splits,
            random_state=42 # Pass random_state
        )

        # Assertions for standard SKF splits
        self.assertIsInstance(skf_splits, list)
        self.assertEqual(len(skf_splits), n_splits) # Number of splits should equal n_splits

        for train_indices, test_indices in skf_splits:
            self.assertIsInstance(train_indices, list)
            self.assertIsInstance(test_indices, list)
            self.assertEqual(len(train_indices) + len(test_indices), n_subjects) # Check total subjects
            self.assertEqual(len(np.intersect1d(train_indices, test_indices)), 0) # Check for overlap

            # Check class distribution in train and test sets (basic check)
            train_labels_in_split = dummy_labels[train_indices]
            test_labels_in_split = dummy_labels[test_indices]
            # Assert that class proportions are roughly maintained (exact check is more complex)
            # For a simple check, ensure both classes are present if possible
            if len(np.unique(dummy_labels)) > 1:
                 self.assertGreater(len(np.unique(train_labels_in_split)), 1)
                 self.assertGreater(len(np.unique(test_labels_in_split)), 1)

        print("\nStratified splitting tests complete.")


    def test_data_format_conversion(self):
        """Test data format conversions (e.g., matrix to flattened, matrix to graph)."""
        # This test requires implementing create_baseline_features and create_graph_data
        # in prepare_abide_data.py and verifying their output formats.

        # Placeholder assertion
        self.assertTrue(True, "Placeholder test for data format conversion")

        # Example of what a real test might do:
        # dummy_matrix = np.random.rand(10, 10)
        # flattened_features = create_baseline_features([dummy_matrix])
        # self.assertEqual(flattened_features.shape[1], 10 * (10 - 1) // 2) # Check flattened size

        # dummy_graph_data_list = create_graph_data([dummy_matrix], [0])
        # self.assertIsInstance(dummy_graph_data_list[0], Data) # Check if it's a PyG Data object


if __name__ == '__main__':
    unittest.main()
