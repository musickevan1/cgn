# Comprehensive CGN Project Context Dump

## 1. Project Overview & Objectives

**CGN (Connectome Graph Networks)** is an AI-augmented connectome analysis project that uses graph neural networks and transformers for neuroimaging analysis. The project is in early development phase (v0.1.0) and targets neuroscience researchers working with brain connectivity data.

**Primary Objectives:**
- Apply graph neural networks to brain connectome analysis
- Develop transformer architectures for neuroimaging data
- Support the ABIDE neuroimaging dataset for autism spectrum disorder research
- Provide end-to-end pipeline from raw neuroimaging data to analysis results, focusing on ASD vs Control binary classification
- Enable rapid research prototyping with 1-2 month development timeline

**Target Users:** Neuroscience researchers, computational neuroscientists, AI researchers working on brain data

**Current Status:** Development phase - basic data loading and preprocessing infrastructure complete, ready for Phase 4 (baseline models and GNNs)

## 2. Architecture & Technical Stack

**Core Technology Stack:**
- **Python 3.8-3.10** (primary language)
- **PyTorch ≥2.0.0** (deep learning framework)
- **PyTorch Geometric ≥2.3.0** (graph neural networks)
- **PyTorch Lightning ≥2.0.0** (training framework)
- **Transformers ≥4.30.0** (transformer architectures)
- **Nilearn ≥0.10.0** (neuroimaging preprocessing)
- **NetworkX ≥3.0** (graph analysis)
- **Nibabel ≥5.0.0** (neuroimaging file I/O)

**Architecture Pattern:** Modular research pipeline with clear separation of concerns:
- Data loading and preprocessing
- Connectome construction
- Model architectures (planned)
- Training pipelines (planned)
- Evaluation and visualization (planned)

**Key Dependencies:**
- Scientific computing: NumPy, SciPy, Pandas, Scikit-learn
- Visualization: Matplotlib, Seaborn, Plotly
- Development: Pytest, Black, Flake8, MyPy
- Containerization: Docker with CUDA support

## 3. Project Structure & Organization

```
CGN/
├── src/connectome_analysis/          # Main package
│   ├── data/                        # Data handling modules
│   │   ├── loaders.py              # Dataset loading (ABIDE, ADHD-200)
│   │   ├── preprocessing.py         # Neuroimaging preprocessing
│   │   └── connectome.py           # Connectome construction
│   ├── models/                     # Model architectures (planned)
│   ├── training/                   # Training pipelines (planned)
│   ├── evaluation/                 # Evaluation metrics (planned)
│   ├── utils/                      # Utility functions
│   └── visualization/              # Plotting and visualization
├── data/                           # Data storage
│   ├── raw/                       # Raw downloaded datasets
│   ├── processed/                 # Processed connectomes
│   └── external/                  # External atlases/templates
├── configs/                       # Configuration files
├── scripts/                       # Utility scripts
│   └── data/download_datasets.py  # Data download script
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # Test suite
├── docker/                        # Containerization
├── docs/                          # Documentation
├── results/                       # Experimental results
├── models/                        # Saved model checkpoints
└── prompts/                       # AI development prompts
```

**Configuration Management:**
- `pyproject.toml` and `setup.py` for package configuration
- `environment.yml` for conda environment
- `requirements.txt` for pip dependencies
- Docker configuration for reproducible environments

## 4. Core Components & Functionality

### Data Loading (`src/connectome_analysis/data/loaders.py`)
**Key Classes:**
- `DatasetLoader`: Base class for neuroimaging dataset loading
- `ABIDELoader`: ABIDE (Autism Brain Imaging Data Exchange) dataset

**Key Functions:**
- `create_dataset_loader()`: Factory function for the ABIDE dataset loader
- `get_available_datasets()`: Returns supported datasets (currently ABIDE only)

**Features:**
- Automatic dataset downloading via nilearn
- Phenotypic data loading
- Graceful handling of missing neuroimaging packages

### Preprocessing (`src/connectome_analysis/data/preprocessing.py`)
**Key Classes:**
- `NeuroPreprocessor`: Main preprocessing pipeline
- `QualityControl`: Quality metrics computation

**Key Functions:**
- `preprocess_fmri()`: Standard fMRI preprocessing pipeline
- `extract_roi_time_series()`: ROI time series extraction using atlases
- `compute_motion_metrics()`: Motion quality control
- `compute_signal_metrics()`: Signal quality assessment
- `load_standard_atlases()`: Download standard brain atlases

**Preprocessing Pipeline:**
- Brain masking
- Temporal filtering (high-pass, low-pass)
- Standardization and detrending
- ROI extraction using anatomical atlases

### Connectome Construction (`src/connectome_analysis/data/connectome.py`)
**Key Classes:**
- `ConnectomeBuilder`: Functional connectivity matrix construction
- `MultiModalConnectome`: Multi-modal connectivity combination

**Key Functions:**
- `build_functional_connectome()`: Build connectivity matrices
- `compute_graph_metrics()`: Graph theory metrics calculation
- `combine_modalities()`: Combine functional/structural connectivity
- `extract_connectome_features()`: Feature extraction from connectivity matrices

**Connectivity Methods:**
- Correlation, partial correlation, tangent space
- Graph theory metrics: efficiency, clustering, modularity, small-worldness
- Multi-modal combination strategies

## 5. Current Development State

**Recently Completed:**
- Basic project structure and directory organization
- Data loading infrastructure for ABIDE and ADHD-200 datasets
- Neuroimaging preprocessing pipeline
- Connectome construction utilities
- Graph theory metrics computation
- Docker containerization setup
- Robust error handling for optional dependencies

**Code Quality Status:**
- All Pylance type checking errors resolved
- Robust import handling with graceful degradation
- NetworkX compatibility across versions
- Comprehensive error handling and logging

**Current Phase:** Ready for Phase 4 development (baseline models and initial GNNs)

## 6. Configuration & Environment

**Environment Setup:**
```bash
# Conda environment
conda env create -f environment.yml
conda activate cgn

# Package installation
pip install -e .

# Data download
python scripts/data/download_datasets.py --dataset abide
```

**Docker Support:**
- Base Dockerfile with Python 3.9
- GPU-enabled Dockerfile for CUDA workloads
- Docker Compose for development environment

**Environment Variables:** Currently minimal, expandable for API keys, data paths

## 7. Challenges & Technical Debt

**Current Challenges:**
- Optional dependency management (nilearn, nibabel)
- NetworkX version compatibility for community detection
- Large neuroimaging dataset handling
- Memory efficiency for connectome matrices

**Known Issues:**
- Limited to correlation-based connectivity when nilearn unavailable
- Modularity computation fallbacks for different NetworkX versions
- No GPU acceleration for preprocessing yet

**Technical Debt Areas:**
- Need comprehensive test suite
- Documentation needs expansion
- Performance optimization for large datasets
- Error handling could be more granular

## 8. Development Workflow & Standards

**Code Organization:**
- Modular design with clear separation of concerns
- Type hints throughout codebase
- Comprehensive docstrings
- Graceful handling of optional dependencies

**Planned Standards:**
- Black code formatting
- Flake8 linting
- MyPy type checking
- Pytest testing framework

**Version Control:** Git-based with structured commit messages

## 9. Dependencies & Integrations

**Core Scientific Dependencies:**
- PyTorch ecosystem (torch, torch-geometric, pytorch-lightning)
- Neuroimaging: nilearn, nibabel
- Scientific computing: numpy, scipy, pandas, scikit-learn
- Graph analysis: networkx

**External Data Sources:**
- ABIDE dataset via nilearn
- ADHD-200 dataset via nilearn
- Standard brain atlases (AAL, Schaefer)

**Optional Dependencies:**
- Robust handling when neuroimaging packages unavailable
- Fallback to basic correlation methods

## 10. Future Roadmap & Next Steps

**Phase 4 (Next):** Baseline Models and Initial Graph Neural Networks (Focus on ASD vs Control Classification with ABIDE)
- Implement baseline classifiers (SVM, Random Forest) for ASD vs Control classification
- Basic Graph Convolutional Networks (GCN) for ABIDE data
- Graph Attention Networks (GAT) for ABIDE data

**Phase 5:** Brain Graph Transformer Architecture
- Custom transformer for brain connectivity
- Positional encodings for brain regions
- Multi-head attention for connectivity patterns

**Phase 6:** Training Pipeline and Hyperparameter Optimization
- PyTorch Lightning training modules
- Hyperparameter optimization with Optuna
- Cross-validation strategies

**Phase 7:** Evaluation and Cross-Validation
- Comprehensive evaluation metrics
- Statistical significance testing
- Visualization of results

**Phase 8:** Results Analysis and Publication Materials
- Scientific plots and figures
- Performance comparisons
- Documentation for publication

## 11. Documentation & Resources

**Available Documentation:**
- README.md with quick start guide
- Installation instructions (basic)
- API documentation structure (planned)
- Prompt-based development history in `prompts/`

**Setup Instructions:**
1. Clone repository
2. Create conda environment from `environment.yml`
3. Install package with `pip install -e .`
4. Download data with provided scripts

**Development Prompts:**
- Structured prompt history for AI-assisted development
- Sequential prompts for each development phase
- Future prompts planned for upcoming phases

## 12. Metrics & Monitoring

**Planned Metrics:**
- Model performance (accuracy, F1, AUC)
- Training metrics (loss, convergence)
- Data quality metrics (motion, signal quality)
- Graph theory metrics (efficiency, modularity)

**Logging:** Basic Python logging, expandable for experiment tracking

**Quality Control:**
- Motion metrics computation
- Signal quality assessment
- Connectome quality validation

---

**Generated:** 2025-05-23 03:01 AM (America/Chicago)  
**Project Version:** 0.1.0  
**Development Phase:** Ready for Phase 4 (Baseline Models and GNNs)

This context dump represents the current state of the CGN project as a well-structured neuroimaging analysis framework ready for advanced model development and research applications.
