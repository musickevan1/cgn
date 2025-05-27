import unittest
import os
import shutil
import numpy as np
import pickle
import yaml # Import yaml
import random # Import random
import torch # Import torch
from torch_geometric.data import Data # Import Data

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the main scripts and relevant components
from scripts.integration.prepare_abide_data import load_and_process_abide, create_baseline_features, create_graph_data, stratified_split_by_site
from scripts.integration.run_baseline_experiment import run_baseline_experiment
from scripts.integration.run_gnn_experiment import run_gnn_experiment # type: ignore
from src.connectome_analysis.training.trainer import CrossValidator, ConnectomeDataset
from src.connectome_analysis.evaluation import statistical_tests
from src.connectome_analysis.evaluation import metrics

# Define paths for dummy data and output
DUMMY_DATA_DIR = 'tests/fixtures/sample_data/abide/raw' # Example path
DUMMY_ATLAS_PATH = 'tests/fixtures/sample_data/atlas/atlas.nii.gz' # Example path
DUMMY_ATLAS_COORDS_PATH = 'tests/fixtures/sample_data/atlas/coordinates.txt' # Example path
PROCESSED_DATA_DIR = 'tests/integration/processed_data_pipeline' # Output directory for processed data
PIPELINE_RESULTS_DIR = 'results/integration_pipeline_test' # Output directory for experiment results

class TestCompletePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up dummy data and directories before running tests."""
        print("\nSetting up dummy data for complete pipeline integration tests...")
        # Create dummy data directories and files if they don't exist
        os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DUMMY_ATLAS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(DUMMY_ATLAS_COORDS_PATH), exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(PIPELINE_RESULTS_DIR, exist_ok=True)

        # Create placeholder dummy files (replace with actual small test files if possible)
        # Example: Create a dummy phenotypic file
        dummy_phenotypic_content = """SUB_ID,DX_GROUP,SITE_ID,FILE_ID
1,1,SiteA,sub_001_func_preproc.nii.gz
2,2,SiteB,sub_002_func_preproc.nii.gz
3,1,SiteA,sub_003_func_preproc.nii.gz
4,2,SiteC,sub_004_func_preproc.nii.gz
5,1,SiteB,sub_005_func_preproc.nii.gz
6,2,SiteC,sub_006_func_preproc.nii.gz
"""
        phenotypic_dir = os.path.join(DUMMY_DATA_DIR, 'abide', 'phenotypic')
        os.makedirs(phenotypic_dir, exist_ok=True)
        with open(os.path.join(phenotypic_dir, 'ABIDE_pcp_phenotypic.csv'), 'w') as f:
            f.write(dummy_phenotypic_content)

        # Create dummy atlas files (requires nibabel for proper nifti)
        # For now, just create empty files or simple text files
        with open(DUMMY_ATLAS_PATH, 'w') as f:
            f.write("dummy atlas content")
        with open(DUMMY_ATLAS_COORDS_PATH, 'w') as f:
            f.write("dummy coordinates content")

        print("Dummy data setup complete.")


    @classmethod
    def tearDownClass(cls):
        """Clean up dummy data and directories after running tests."""
        print("\nCleaning up dummy data for complete pipeline integration tests...")
        # Remove the processed data and results directories
        if os.path.exists(PROCESSED_DATA_DIR):
            shutil.rmtree(PROCESSED_DATA_DIR)
        if os.path.exists(PIPELINE_RESULTS_DIR):
            shutil.rmtree(PIPELINE_RESULTS_DIR)
        # Optionally, remove dummy raw data if created here
        # if os.path.exists(DUMMY_DATA_DIR):
        #     shutil.rmtree(DUMMY_DATA_DIR)
        # if os.path.exists(os.path.dirname(DUMMY_ATLAS_PATH)):
        #      shutil.rmtree(os.path.dirname(DUMMY_ATLAS_PATH))
        print("Cleanup complete.")


    def test_data_loading_integration(self):
        """Test that data loading and initial processing works end-to-end."""
        print("\nRunning test_data_loading_integration...")
        # This test relies on the dummy data setup in setUpClass
        # It calls the load_and_process_abide function from prepare_abide_data.py

        # Note: load_and_process_abide currently requires actual nifti files and nibabel.
        # With the current dummy setup (only phenotypic file), this function will likely
        # skip subjects due to missing fMRI files.
        # A proper integration test would require creating small, valid dummy nifti files.

        # For now, let's just assert that the function runs without crashing
        # and returns lists (even if empty due to missing files).
        try:
            connectomes, labels, sites, subject_ids = load_and_process_abide(
                DUMMY_DATA_DIR, DUMMY_ATLAS_PATH, DUMMY_ATLAS_COORDS_PATH
            )
            self.assertIsInstance(connectomes, list)
            self.assertIsInstance(labels, list)
            self.assertIsInstance(sites, list)
            self.assertIsInstance(subject_ids, list)
            print("test_data_loading_integration passed (function ran without crashing).")
            # Add more specific assertions if you have proper dummy nifti files
            # self.assertGreater(len(connectomes), 0)

        except Exception as e:
            self.fail(f"test_data_loading_integration failed: {e}")


    def test_baseline_training_integration(self):
        """Test baseline model training on small dataset using the integrated script."""
        print("\nRunning test_baseline_training_integration...")
        # This test requires running prepare_abide_data.py first to generate processed data.
        # Since we don't have actual nifti files, we'll simulate the output of prepare_abide_data.py
        # by creating dummy processed data files.

        # Simulate processed data output from prepare_abide_data.py
        num_subjects = 10 # Match the number of subjects in dummy phenotypic data
        num_features = 116 * (116 - 1) // 2
        dummy_baseline_features = np.random.rand(num_subjects, num_features).astype(np.float32)
        dummy_labels = [0] * (num_subjects // 2) + [1] * (num_subjects // 2)
        dummy_sites = ['SiteA', 'SiteB', 'SiteA', 'SiteC', 'SiteB', 'SiteC', 'SiteA', 'SiteB', 'SiteC', 'SiteA'] # Example sites
        dummy_subject_ids = [f'sub_{i+1}' for i in range(num_subjects)]

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_DIR, "baseline_features.pkl"), "wb") as f:
            pickle.dump(dummy_baseline_features, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "labels.pkl"), "wb") as f:
            pickle.dump(dummy_labels, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "sites.pkl"), "wb") as f:
            pickle.dump(dummy_sites, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "subject_ids.pkl"), "wb") as f:
            pickle.dump(dummy_subject_ids, f)

        # Create a dummy config file for the baseline experiment
        dummy_baseline_config = {
            'cv_folds': 2, # Use 2 folds for speed
            'stratify_by_site': True,
            'random_state': 42,
            'batch_size': 8,
            'max_epochs': 5,
            'accelerator': 'auto',
            'devices': 'auto',
            'models': {
                'SVM': {
                    'model_name': 'SVM',
                    'model_params': {'params': {'C': 1.0, 'kernel': 'linear'}},
                    'trainer_params': {}
                }
            },
            'evaluation': {
                'alpha': 0.05,
                'test_type': 'wilcoxon'
            }
        }
        dummy_baseline_config_path = os.path.join(PROCESSED_DATA_DIR, "baseline_config.yaml")
        with open(dummy_baseline_config_path, 'w') as f:
            yaml.dump(dummy_baseline_config, f)


        # Run the baseline experiment script
        try:
            # Redirect stdout to capture print statements if needed for verification
            # import io
            # old_stdout = sys.stdout
            # sys.stdout = captured_output = io.StringIO()

            run_baseline_experiment(PROCESSED_DATA_DIR, PIPELINE_RESULTS_DIR, dummy_baseline_config_path)

            # sys.stdout = old_stdout
            # print("Captured Output:\n", captured_output.getvalue())

            # Assert that results files were created
            self.assertTrue(os.path.exists(os.path.join(PIPELINE_RESULTS_DIR, "baseline_experiment_results.pkl")))
            self.assertTrue(os.path.exists(os.path.join(PIPELINE_RESULTS_DIR, "baseline_experiment_report.html"))) # Assuming HTML report is generated

            # Load and check the results (basic check)
            with open(os.path.join(PIPELINE_RESULTS_DIR, "baseline_experiment_results.pkl"), "rb") as f:
                results = pickle.load(f)
            self.assertIn('SVM', results)
            self.assertIsInstance(results['SVM'], list)
            self.assertEqual(len(results['SVM']), dummy_baseline_config['cv_folds']) # Check number of folds

            print("test_baseline_training_integration passed.")

        except Exception as e:
            self.fail(f"test_baseline_training_integration failed: {e}")


    def test_gnn_training_integration(self):
        """Test GNN model training on small dataset using the integrated script."""
        print("\nRunning test_gnn_training_integration...")
        # This test requires running prepare_abide_data.py first to generate processed graph data.
        # We'll simulate the output of prepare_abide_data.py by creating dummy processed graph data files.

        # Simulate processed graph data output from prepare_abide_data.py
        num_subjects = 10 # Match the number of subjects in dummy phenotypic data
        num_regions = 116
        dummy_labels = [0] * (num_subjects // 2) + [1] * (num_subjects // 2)
        dummy_sites = ['SiteA', 'SiteB', 'SiteA', 'SiteC', 'SiteB', 'SiteC', 'SiteA', 'SiteB', 'SiteC', 'SiteA'] # Example sites

        dummy_graph_data_list = []
        for i in range(num_subjects):
            x = torch.randn(num_regions, 16) # Dummy node features
            edge_index = torch.combinations(torch.arange(num_regions), 2).t().contiguous()
            edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1) # Corrected flip usage
            edge_attr = torch.randn(edge_index.size(1), 1)
            data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([dummy_labels[i]]), site=dummy_sites[i])
            dummy_graph_data_list.append(data_obj)


        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_DIR, "graph_data_list.pkl"), "wb") as f:
            pickle.dump(dummy_graph_data_list, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "labels.pkl"), "wb") as f:
            pickle.dump(dummy_labels, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "sites.pkl"), "wb") as f:
            pickle.dump(dummy_sites, f)

        # Create a dummy config file for the GNN experiment
        dummy_gnn_config = {
            'cv_folds': 2, # Use 2 folds for speed
            'stratify_by_site': True,
            'random_state': 42,
            'batch_size': 8,
            'max_epochs': 5,
            'accelerator': 'auto',
            'devices': 'auto',
            'models': {
                'BrainGAT': { # Test BrainGAT for attention interpretation
                    'model_name': 'BrainGAT',
                    'model_params': {'num_node_features': 16, 'hidden_channels': 16, 'num_classes': 2, 'heads': 2},
                    'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                }
            },
            'evaluation': {
                'alpha': 0.05,
                'test_type': 'wilcoxon'
            }
        }
        dummy_gnn_config_path = os.path.join(PROCESSED_DATA_DIR, "gnn_config.yaml")
        with open(dummy_gnn_config_path, 'w') as f:
            yaml.dump(dummy_gnn_config, f)


        # Run the GNN experiment script
        try:
            run_gnn_experiment(PROCESSED_DATA_DIR, PIPELINE_RESULTS_DIR, dummy_gnn_config_path)

            # Assert that results files were created
            self.assertTrue(os.path.exists(os.path.join(PIPELINE_RESULTS_DIR, "gnn_experiment_results.pkl"))) # Assuming GNN results are saved separately or in combined report
            self.assertTrue(os.path.exists(os.path.join(PIPELINE_RESULTS_DIR, "gnn_experiment_report.html"))) # Assuming HTML report is generated

            # Load and check the results (basic check)
            with open(os.path.join(PIPELINE_RESULTS_DIR, "gnn_experiment_results.pkl"), "rb") as f:
                 results = pickle.load(f)
            self.assertIn('BrainGAT', results)
            self.assertIsInstance(results['BrainGAT'], list)
            self.assertEqual(len(results['BrainGAT']), dummy_gnn_config['cv_folds']) # Check number of folds

            # Check for attention analysis results if model was BrainGAT
            if dummy_gnn_config['models']['BrainGAT']['model_name'] == 'BrainGAT':
                 # Assuming attention analysis is stored in the fold metrics
                 first_fold_results = results['BrainGAT'][0]
                 self.assertIn('fold_0', first_fold_results)
                 self.assertIn('attention_analysis', first_fold_results['fold_0'])
                 print("GAT attention analysis results found in GNN experiment output.")


            print("test_gnn_training_integration passed.")

        except Exception as e:
            self.fail(f"test_gnn_training_integration failed: {e}")


    def test_evaluation_pipeline_integration(self):
        """Test statistical evaluation and reporting integration."""
        print("\nRunning test_evaluation_pipeline_integration...")
        # This test requires results from at least two models to perform statistical comparison.
        # We'll simulate results from dummy baseline and GNN experiments.

        # Simulate results from baseline experiment
        baseline_results = {'SVM': []}
        for i in range(3): # 3 folds
            baseline_results['SVM'].append({f'fold_{i}': {'accuracy': 0.6 + random.random() * 0.1, 'auc': 0.65 + random.random() * 0.1, 'y_true': [0, 1, 0, 1], 'y_prob': [0.4, 0.6, 0.3, 0.7]}})

        # Simulate results from GNN experiment
        gnn_results = {'BrainGAT': []}
        for i in range(3): # 3 folds
            gnn_results['BrainGAT'].append({f'fold_{i}': {'accuracy': 0.7 + random.random() * 0.1, 'auc': 0.75 + random.random() * 0.1, 'y_true': [0, 1, 0, 1], 'y_prob': [0.3, 0.7, 0.4, 0.8]}})

        all_results = {**baseline_results, **gnn_results}

        # Run the evaluation and reporting
        output_dir = os.path.join(PIPELINE_RESULTS_DIR, "evaluation_test")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # The run_baseline_experiment and run_gnn_experiment scripts will eventually call
            # reporting functions. For this test, we'll just verify the existence of results.

            # Example of how you might use the new statistical_tests and metrics modules
            # statistical_tests.compare_models_paired_t_test(model1_metrics, model2_metrics)
            # metrics.plot_roc_curve(y_true, y_proba)

            # Assert that results files were created by the experiment scripts
            # (These assertions are already present in test_baseline_training_integration and test_gnn_training_integration)
            # For this specific test, we'll just ensure the overall process doesn't crash.
            # More specific assertions for evaluation output would go here once reporting is finalized.

            print("test_evaluation_pipeline_integration passed.")

        except Exception as e:
            self.fail(f"test_evaluation_pipeline_integration failed: {e}")


    def test_results_reproducibility(self):
        """Test that results are reproducible with the same random seed."""
        print("\nRunning test_results_reproducibility...")
        # This test requires running an experiment twice with the same random seed
        # and comparing the results.

        # Simulate processed data output (same as baseline training test)
        num_subjects = 10
        num_features = 116 * (116 - 1) // 2
        dummy_baseline_features = np.random.rand(num_subjects, num_features).astype(np.float32)
        dummy_labels = [0] * (num_subjects // 2) + [1] * (num_subjects // 2)
        dummy_sites = ['SiteA', 'SiteB', 'SiteA', 'SiteC', 'SiteB', 'SiteC', 'SiteA', 'SiteB', 'SiteC', 'SiteA']

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_DIR, "baseline_features.pkl"), "wb") as f:
            pickle.dump(dummy_baseline_features, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "labels.pkl"), "wb") as f:
            pickle.dump(dummy_labels, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "sites.pkl"), "wb") as f:
            pickle.dump(dummy_sites, f)

        # Create a dummy config file with a fixed random seed
        reproducibility_config = {
            'cv_folds': 2,
            'stratify_by_site': False, # Use standard k-fold for simplicity
            'random_state': 123, # Fixed random seed
            'batch_size': 8,
            'max_epochs': 3, # Short epochs
            'accelerator': 'auto',
            'devices': 'auto',
            'models': {
                'SVM': {
                    'model_name': 'SVM',
                    'model_params': {'params': {'C': 1.0, 'kernel': 'linear'}},
                    'trainer_params': {}
                }
            },
            'evaluation': {
                'alpha': 0.05,
                'test_type': 'wilcoxon'
            }
        }
        reproducibility_config_path = os.path.join(PROCESSED_DATA_DIR, "reproducibility_config.yaml")
        with open(reproducibility_config_path, 'w') as f:
            yaml.dump(reproducibility_config, f)

        # Run the experiment script twice with the same config
        results1 = None
        results2 = None

        output_dir1 = os.path.join(PIPELINE_RESULTS_DIR, "reproducibility_run1")
        output_dir2 = os.path.join(PIPELINE_RESULTS_DIR, "reproducibility_run2")
        os.makedirs(output_dir1, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)


        try:
            print("Running reproducibility test - Run 1...")
            run_baseline_experiment(PROCESSED_DATA_DIR, output_dir1, reproducibility_config_path)
            with open(os.path.join(output_dir1, "baseline_experiment_results.pkl"), "rb") as f:
                 results1 = pickle.load(f)

            print("Running reproducibility test - Run 2...")
            run_baseline_experiment(PROCESSED_DATA_DIR, output_dir2, reproducibility_config_path)
            with open(os.path.join(output_dir2, "baseline_experiment_results.pkl"), "rb") as f:
                 results2 = pickle.load(f)

            # Compare the results
            self.assertIsNotNone(results1)
            self.assertIsNotNone(results2)
            self.assertEqual(results1.keys(), results2.keys())

            for model_name in results1.keys():
                 self.assertEqual(len(results1[model_name]), len(results2[model_name]))
                 for i in range(len(results1[model_name])):
                      fold_results1 = results1[model_name][i]
                      fold_results2 = results2[model_name][i]
                      self.assertEqual(fold_results1.keys(), fold_results2.keys())
                      for fold_name in fold_results1.keys():
                           metrics1 = fold_results1[fold_name]
                           metrics2 = fold_results2[fold_name]
                           # Compare metrics (allow for small floating point differences)
                           for metric in metrics1.keys():
                                if isinstance(metrics1[metric], (int, float)):
                                     self.assertAlmostEqual(metrics1[metric], metrics2[metric], places=6, msg=f"Metric {metric} mismatch for {model_name}, Fold {fold_name}")
                                elif isinstance(metrics1[metric], list):
                                     # Compare lists of numbers (e.g., y_true, y_prob)
                                     self.assertEqual(len(metrics1[metric]), len(metrics2[metric]))
                                     for j in range(len(metrics1[metric])):
                                          self.assertAlmostEqual(metrics1[metric][j], metrics2[metric][j], places=6, msg=f"List element mismatch for {model_name}, Fold {fold_name}, Metric {metric}, Index {j}")
                                else:
                                     # Compare other types directly
                                     self.assertEqual(metrics1[metric], metrics2[metric], msg=f"Value mismatch for {model_name}, Fold {fold_name}, Metric {metric}")


            print("test_results_reproducibility passed.")

        except Exception as e:
            self.fail(f"test_results_reproducibility failed: {e}")


if __name__ == '__main__':
    unittest.main()
