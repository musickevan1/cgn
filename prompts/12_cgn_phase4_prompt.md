# CGN Phase 4: Baseline Models and Initial Graph Neural Networks

## Project Status Update
- **Current Phase**: Phase 4 Implementation
- **Previous Completion**: Data infrastructure (loaders, preprocessing, connectome construction)
- **Pylance Issues**: Recently addressed type checking errors
- **Ready For**: Baseline model implementation and initial GNN development

## Phase 4 Objectives

### Primary Goals
1. **Implement Baseline Classifiers**
   - Support Vector Machines (SVM)
   - Random Forest
   - Logistic Regression
   - Performance benchmarking on ABIDE/ADHD-200

2. **Develop Initial Graph Neural Networks**
   - Graph Convolutional Networks (GCN)
   - Graph Attention Networks (GAT)
   - Graph classification for connectome analysis

3. **Establish Training Pipeline**
   - Cross-validation strategies for neuroimaging
   - Performance metrics and evaluation
   - Model comparison framework

## Technical Requirements

### Architecture Implementation
```
src/connectome_analysis/models/
├── baseline.py          # Traditional ML models (EXISTING - needs expansion)
├── gnn_models.py       # GCN, GAT implementations (NEW)
├── graph_utils.py      # Graph construction utilities (NEW)
└── __init__.py         # Model factory functions (UPDATE)

src/connectome_analysis/training/
├── trainer.py          # Training orchestration (EXISTING - needs GNN support)
├── cross_validation.py # CV strategies for neuroimaging (NEW)
└── experiment.py       # Experiment management (NEW)
```

### Key Components Needed

#### 1. Enhanced Baseline Models (`models/baseline.py`)
- Expand existing baseline classifiers
- Add feature selection methods
- Implement proper cross-validation
- Support for connectivity matrix features

#### 2. Graph Neural Networks (`models/gnn_models.py`)
```python
class BrainGCN(torch.nn.Module):
    """Graph Convolutional Network for brain connectomes"""
    
class BrainGAT(torch.nn.Module):
    """Graph Attention Network for brain connectivity analysis"""
    
class ConnectomeClassifier(torch.nn.Module):
    """Wrapper for graph-level classification tasks"""
```

#### 3. Graph Construction (`models/graph_utils.py`)
```python
def connectivity_to_pyg_data(connectivity_matrix, labels, node_features=None):
    """Convert connectivity matrix to PyTorch Geometric Data object"""
    
def create_brain_graph_dataset(connectomes, labels, atlas_info):
    """Create graph dataset from connectome matrices"""
```

#### 4. Training Infrastructure (`training/`)
- Cross-validation for neuroimaging data (leave-one-site-out)
- Hyperparameter optimization
- Model comparison and statistical testing
- Experiment tracking and reproducibility

## Implementation Priority

### Week 1: Enhanced Baselines
1. **Expand `models/baseline.py`**
   - Add comprehensive feature extraction from connectomes
   - Implement feature selection (mutual information, ANOVA F-test)
   - Add ensemble methods (Voting, Bagging)
   - Include proper preprocessing pipelines

2. **Create `training/cross_validation.py`**
   - Leave-one-site-out CV for multi-site neuroimaging
   - Stratified CV respecting demographic variables
   - Performance metrics specific to neuroimaging

### Week 2: Initial GNN Implementation
1. **Create `models/gnn_models.py`**
   - Basic GCN implementation using PyTorch Geometric
   - Graph pooling methods (global mean, attention-based)
   - Binary classification head for ASD vs Control

2. **Create `models/graph_utils.py`**
   - Convert connectivity matrices to graph format
   - Handle different thresholding strategies
   - Node feature engineering (anatomical properties)

### Week 3: Training Pipeline Integration
1. **Update `training/trainer.py`**
   - Support for PyTorch Geometric data loaders
   - GNN-specific training loops
   - Gradient accumulation for small batch sizes

2. **Create `training/experiment.py`**
   - Experiment configuration management
   - Results logging and comparison
   - Statistical significance testing

### Week 4: Advanced GNN Features
1. **Implement Graph Attention Networks**
   - Multi-head attention for brain regions
   - Interpretability through attention weights
   - Comparison with GCN performance

2. **Add Advanced Features**
   - Dynamic connectivity (temporal graphs)
   - Multi-modal integration (structural + functional)
   - Hierarchical graph representations

## Datasets and Benchmarks

### Primary Datasets
- **ABIDE I & II**: Autism spectrum disorder classification
- **ADHD-200**: ADHD subtype classification
- **Synthetic Data**: For method validation and testing

### Benchmark Targets
- **ABIDE Classification**: Target 70-75% accuracy (state-of-the-art)
- **Cross-site Generalization**: Robust performance across scanning sites
- **Interpretability**: Identify important brain connections

## Code Quality Standards

### Type Safety
- Resolve all Pylance warnings before implementation
- Add comprehensive type hints for all new functions
- Use proper Union types for flexible returns

### Testing
- Unit tests for all graph construction functions
- Integration tests for training pipelines
- Validation tests on small synthetic datasets

### Documentation
- Comprehensive docstrings with neuroimaging context
- Usage examples for each model class
- Performance benchmarking documentation

## Technical Specifications

### Dependencies to Add
```python
# Add to requirements.txt or environment.yml
torch-geometric>=2.3.0
optuna>=3.0.0           # Hyperparameter optimization
wandb>=0.15.0           # Experiment tracking (optional)
plotly>=5.0.0           # Interactive visualizations
```

### Configuration Management
```yaml
# configs/phase4_config.yaml
models:
  baseline:
    svm:
      kernel: ['rbf', 'linear']
      C: [0.01, 0.1, 1.0, 10.0]
    random_forest:
      n_estimators: [100, 200, 500]
      max_depth: [10, 20, None]
  
  gnn:
    gcn:
      hidden_dim: [64, 128, 256]
      num_layers: [2, 3, 4]
      dropout: [0.2, 0.5]
    
training:
  cross_validation:
    strategy: 'leave_one_site_out'
    n_splits: 5
  
  optimization:
    n_trials: 100
    timeout: 3600  # 1 hour
```

## Expected Deliverables

### Code Components
1. Enhanced baseline model suite with proper CV
2. Working GCN and GAT implementations
3. Comprehensive training and evaluation pipeline
4. Graph construction and preprocessing utilities

### Performance Benchmarks
1. Baseline performance on ABIDE dataset
2. GNN performance comparison with literature
3. Cross-site generalization analysis
4. Computational efficiency metrics

### Documentation
1. Model architecture documentation
2. Usage tutorials and examples
3. Performance analysis and interpretation
4. Troubleshooting guide

## Success Metrics

### Technical Milestones
- [ ] All Pylance errors resolved
- [ ] Baseline models achieve >65% accuracy on ABIDE
- [ ] GCN implementation trains successfully
- [ ] Cross-validation pipeline works end-to-end
- [ ] Results match or exceed literature benchmarks

### Research Impact
- [ ] Novel insights into brain connectivity patterns
- [ ] Interpretable model explanations
- [ ] Reproducible research pipeline
- [ ] Contribution to open science in neuroimaging

## Next Steps After Phase 4

### Phase 5 Preview: Brain Graph Transformers
- Custom transformer architecture for brain graphs
- Positional encodings for anatomical regions
- Multi-scale attention mechanisms
- Integration with Phase 4 baseline comparisons

## Development Notes

### Neuroimaging Considerations
- Handle site effects in multi-site datasets
- Proper motion artifact control
- Atlas-based vs. data-driven parcellations
- Statistical power and multiple comparisons

### Machine Learning Best Practices
- Prevent data leakage in cross-validation
- Proper hyperparameter optimization
- Model interpretability and explainability
- Computational efficiency for large datasets

### Integration Points
- Use existing data loading infrastructure
- Leverage preprocessing pipeline
- Build on connectome construction utilities
- Maintain compatibility with Docker environment

---

**Prompt Usage**: Use this prompt to guide Phase 4 development of the CGN project. Focus on implementing robust baseline models first, then gradually introduce GNN components while maintaining scientific rigor and code quality standards.
