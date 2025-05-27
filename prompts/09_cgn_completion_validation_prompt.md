# CGN Phase 4 Completion & Validation: Core Functions and End-to-End Testing

## Project Context
You have successfully completed **80% of the CGN Phase 4 integration**:
- ✅ Complete integration infrastructure (scripts, notebooks, tests)
- ✅ Core pipeline components (data preparation, experiment workflows)
- ✅ Enhanced cross-validation framework
- ✅ Validation and benchmarking structure

**Current Objective:** Complete the remaining **20%** to create a fully functional, scientifically rigorous research pipeline that can generate publishable results on ABIDE data.

## Completion Priority Tasks

### 1. Complete Core Evaluation Functions
File: `src/connectome_analysis/evaluation/metrics.py`

#### Implement `statistical_significance_test()`
**Purpose:** Rigorous statistical comparison between model performances

**Requirements:**
```python
def statistical_significance_test(results_dict, test_type='wilcoxon', alpha=0.05):
    """
    Test for significant differences between model performances using appropriate statistical tests
    
    Args:
        results_dict: Dict with model names as keys, cross-validation scores as values
                     Example: {'SVM': [0.65, 0.67, 0.63, 0.69, 0.64], 
                              'GCN': [0.72, 0.74, 0.71, 0.73, 0.70]}
        test_type: Statistical test to use
                  - 'wilcoxon': Wilcoxon signed-rank test (non-parametric, paired)
                  - 'ttest': Paired t-test (parametric, assumes normality)
                  - 'mcnemar': McNemar's test (for classification, requires predictions)
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict containing:
        - pairwise_results: All pairwise comparisons with p-values and effect sizes
        - significant_pairs: Pairs with significant differences
        - ranking: Models ranked by mean performance
        - summary_table: Formatted results table
    """
    
    # Implementation should include:
    # 1. Pairwise statistical tests between all model combinations
    # 2. Multiple comparison correction (Bonferroni or Holm-Bonferroni)
    # 3. Effect size calculation (Cohen's d for t-test, rank-biserial for Wilcoxon)
    # 4. Confidence intervals for performance differences
    # 5. Summary statistics (mean, std, CI for each model)
    # 6. Formatted output suitable for research reporting
```

#### Implement `interpret_gat_attention()`
**Purpose:** Extract and analyze GAT attention weights for neuroscience interpretability

**Requirements:**
```python
def interpret_gat_attention(model, data_loader, atlas_info=None, top_k=20):
    """
    Extract and analyze attention weights from trained GAT model
    
    Args:
        model: Trained BrainGAT model from gnn.py
        data_loader: PyTorch DataLoader containing graph data
        atlas_info: Dict with atlas information
                   {'region_names': [...], 'coordinates': [...], 'networks': [...]}
        top_k: Number of top connections to highlight
    
    Returns:
        Dict containing:
        - attention_matrices: Average attention weights across subjects/heads
        - top_connections: Most attended connections with brain region names
        - network_attention: Attention patterns by brain networks (DMN, SN, etc.)
        - subject_variability: How attention patterns vary across subjects
        - visualization_data: Data formatted for brain visualization
    """
    
    # Implementation should include:
    # 1. Extract attention weights from all GAT layers and heads
    # 2. Average across subjects and multiple attention heads
    # 3. Identify most consistently attended connections
    # 4. Map connections to brain region names using atlas
    # 5. Analyze attention by known brain networks
    # 6. Compute statistical significance of attention patterns
    # 7. Format data for brain visualization tools
```

#### Implement `plot_roc_curves()`
**Purpose:** Create publication-quality ROC curve comparisons

**Requirements:**
```python
def plot_roc_curves(results_dict, save_path=None, title="Model Comparison"):
    """
    Plot ROC curves comparing multiple models with statistical annotations
    
    Args:
        results_dict: Dict with model results containing y_true, y_prob for each fold
                     Example: {'SVM': {'fold_0': {'y_true': [...], 'y_prob': [...]}, ...}}
        save_path: Path to save the figure (optional)
        title: Plot title
    
    Returns:
        matplotlib.figure.Figure: The generated figure
        Dict: AUC statistics (mean, std, CI) for each model
    """
    
    # Implementation should include:
    # 1. Plot mean ROC curve with confidence intervals for each model
    # 2. Calculate and display mean AUC ± std for each model
    # 3. Statistical testing between AUC values
    # 4. Publication-quality formatting (proper colors, legend, labels)
    # 5. Annotations showing significant differences between models
    # 6. Both individual fold curves (faded) and mean curves (bold)
```

### 2. Fix trainer.py Implementation Issues
File: `src/connectome_analysis/training/trainer.py`

#### Complete the `CrossValidator` class
**Focus Areas:**
```python
def create_splits(self, data, labels, sites):
    """
    Create site-stratified cross-validation splits ensuring:
    1. No subject appears in both train and test
    2. Sites are distributed across folds as evenly as possible
    3. Class balance is maintained within each fold
    4. Handles ABIDE's multi-site structure properly
    """
    
def evaluate_baseline_model(self, model_class, X, y, sites, **model_params):
    """
    Complete cross-validation for sklearn baseline models:
    1. Proper feature scaling within each fold (fit on train, transform test)
    2. Hyperparameter optimization using inner CV loop
    3. Collect predictions and probabilities for statistical testing
    4. Handle class imbalance appropriately
    """
    
def evaluate_gnn_model(self, model_class, graph_data, **model_params):
    """
    Complete cross-validation for PyTorch GNN models:
    1. Convert graph data to PyTorch Geometric format
    2. Handle graph batching properly
    3. Integrate with ConnectomeTrainer for training
    4. Extract attention weights (for GAT models)
    """
```

### 3. Create Minimal Working Example
File: `scripts/validation/minimal_pipeline_test.py`

**Purpose:** Prove the entire pipeline works end-to-end with a small subset of ABIDE data

**Requirements:**
```python
#!/usr/bin/env python3
"""
Minimal working example demonstrating the complete CGN pipeline
Tests the integration with a small subset of ABIDE data
"""

def minimal_pipeline_test():
    """
    Complete end-to-end test with ~50 ABIDE subjects:
    1. Load small ABIDE subset using existing data loaders
    2. Create connectivity matrices using existing connectome builder
    3. Train one baseline model (SVM) and one GNN model (GCN)
    4. Evaluate using proper cross-validation
    5. Generate statistical comparison
    6. Create simple visualization
    7. Report results and timing
    """
    
    # Test workflow:
    # Data Loading (5-10 subjects per class for speed)
    # → Connectivity Matrix Creation
    # → Feature Preparation (flattened for SVM, graphs for GCN)
    # → Model Training with CV
    # → Statistical Comparison
    # → Results Report
    
    # Success criteria:
    # - No errors in pipeline execution
    # - Models achieve >50% accuracy (above chance)
    # - Statistical comparison completes
    # - Results are reproducible across runs
```

### 4. Complete Implementation of Key Script Placeholders
Priority focus on making the integration scripts actually functional:

#### `scripts/integration/prepare_abide_data.py`
**Complete these functions:**
```python
def create_baseline_features(connectomes):
    """
    Convert connectivity matrices to feature vectors for sklearn models
    - Extract upper triangular matrix (avoid redundancy)
    - Apply Fisher z-transformation to correlations
    - Handle missing values and infinite correlations
    - Feature scaling and selection options
    """

def create_graph_data(connectomes, labels, atlas_coords=None):
    """
    Convert connectivity matrices to PyTorch Geometric Data objects
    - Create edge_index and edge_attr from connectivity matrices
    - Handle thresholding for sparse graphs
    - Add node features if available (region coordinates, volumes)
    - Batch multiple subjects into DataLoader format
    """

def stratified_split_by_site(data, labels, sites, test_size=0.2, val_size=0.1):
    """
    Create splits that respect ABIDE site structure
    - Ensure no site appears in both train and test
    - Maintain class balance across splits
    - Create both simple splits and CV fold indices
    """
```

#### `scripts/integration/run_baseline_experiment.py`
**Make this script fully executable:**
```python
def main():
    """
    Complete baseline experiment that actually runs:
    1. Load prepared ABIDE data from prepare_abide_data.py
    2. Initialize all baseline models with hyperparameter grids
    3. Run cross-validation using enhanced CrossValidator
    4. Generate statistical comparisons using completed metrics
    5. Save results in organized format
    6. Create summary report with performance metrics
    """
```

### 5. Integration Testing
File: `tests/integration/test_complete_pipeline.py`

**Purpose:** Automated testing to ensure pipeline reliability

**Requirements:**
```python
class TestCompletePipeline:
    def test_data_loading_integration(self):
        """Test that data loading works with actual ABIDE data"""
        
    def test_baseline_training_integration(self):
        """Test baseline model training on small dataset"""
        
    def test_gnn_training_integration(self):
        """Test GNN model training on small dataset"""
        
    def test_evaluation_pipeline_integration(self):
        """Test statistical evaluation and reporting"""
        
    def test_results_reproducibility(self):
        """Test that results are reproducible with same random seed"""
```

## Implementation Guidelines

### Scientific Rigor Requirements
- **Statistical Testing:** Use appropriate tests with multiple comparison correction
- **Cross-Validation:** Strict site stratification, no data leakage
- **Reproducibility:** Set random seeds, document software versions
- **Interpretability:** GAT attention analysis must be neurologically meaningful
- **Benchmarking:** Compare against published ABIDE results

### Code Quality Standards
- **Error Handling:** Graceful failure with informative messages
- **Documentation:** Clear docstrings with examples
- **Type Hints:** Maintain consistency with existing codebase
- **Testing:** Unit tests for critical statistical functions
- **Performance:** Efficient handling of large connectivity matrices

### Expected Performance Targets
- **Baseline Models:** 60-65% accuracy (matching literature)
- **GNN Models:** 70-75% accuracy (state-of-the-art target)
- **Statistical Significance:** p < 0.05 for GNN vs baseline improvement
- **Reproducibility:** < 2% variance across runs with same seed

## Deliverables for Completion

### Core Functions (Priority 1)
1. **Complete evaluation functions** in `metrics.py`
2. **Fixed CrossValidator** implementation in `trainer.py`
3. **Minimal working example** proving end-to-end functionality

### Integration Scripts (Priority 2)
1. **Functional data preparation** script
2. **Working baseline experiment** script
3. **Working GNN experiment** script

### Validation Framework (Priority 3)
1. **Integration tests** ensuring pipeline reliability
2. **Performance benchmarks** validating against literature
3. **Results reporting** with statistical summaries

## Success Criteria

### Technical Success
- ✅ Complete pipeline runs without errors on ABIDE data
- ✅ All statistical functions work correctly with proper significance testing
- ✅ Models achieve literature-comparable performance
- ✅ Results are reproducible across multiple runs

### Scientific Success
- ✅ GAT attention patterns highlight neurologically plausible connections
- ✅ Statistical comparisons between models are rigorous and interpretable
- ✅ Cross-validation properly handles ABIDE multi-site structure
- ✅ Results ready for scientific publication

### Research Impact
- ✅ Pipeline supports rapid experimentation by neuroscience researchers
- ✅ Code is well-documented and can be easily extended
- ✅ Foundation is solid for Phase 5 Brain Graph Transformer development
- ✅ Results contribute to understanding of autism connectomics

## Timeline for Completion
- **Days 1-2:** Complete core evaluation functions (`metrics.py`)
- **Days 3-4:** Fix trainer.py and create minimal working example
- **Days 5-6:** Complete integration scripts and run full experiments
- **Days 7:** Validation, testing, and documentation cleanup

Focus on creating a robust, scientifically valid pipeline that demonstrates clear improvements of GNN methods over traditional approaches while maintaining the rigor needed for peer-reviewed research.