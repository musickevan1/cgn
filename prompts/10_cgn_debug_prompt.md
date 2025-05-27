# CGN Project Systematic Debugging Prompt

## Context
You are debugging the CGN (Connectome Graph Networks) project - an AI-augmented connectome analysis system for neuroscience research. The project is in Phase 4 (Baseline Models and Initial GNNs) and uses PyTorch, PyTorch Geometric, and neuroimaging libraries.

## Project Structure to Debug
```
CGN/
├── src/connectome_analysis/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # Dataset loading (ABIDE, ADHD-200)
│   │   ├── preprocessing.py     # fMRI preprocessing pipeline
│   │   └── connectome.py       # Connectome construction
│   ├── models/                 # Model architectures (Phase 4)
│   ├── training/               # Training pipelines
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Utility functions
├── setup.py / pyproject.toml   # Package configuration
├── requirements.txt            # Dependencies
├── environment.yml             # Conda environment
└── tests/                      # Test suite
```

## Debugging Strategy

### Phase 1: Environment and Dependencies
**Check these first:**

1. **Verify Python environment and package installations:**
   ```bash
   python --version  # Should be 3.8-3.10
   conda list | grep -E "(torch|nibabel|nilearn|networkx)"
   pip show torch torch-geometric pytorch-lightning
   ```

2. **Test core imports systematically:**
   ```python
   # Test each critical import separately
   import torch
   import torch_geometric
   import pytorch_lightning as pl
   import nilearn  # May be optional
   import nibabel  # May be optional
   import networkx as nx
   import numpy as np
   import pandas as pd
   import sklearn
   ```

3. **Check for version conflicts:**
   - PyTorch ≥2.0.0 compatibility with PyTorch Geometric ≥2.3.0
   - NetworkX ≥3.0 community detection methods
   - Optional neuroimaging dependencies (nilearn, nibabel)

### Phase 2: Core Module Analysis
**Go through each module systematically:**

#### A. Check `src/connectome_analysis/__init__.py`
- Verify package initialization
- Check for circular imports
- Ensure proper module exposure

#### B. Debug `src/connectome_analysis/data/loaders.py`
**Look for these common issues:**
- Import errors with optional dependencies (nilearn, nibabel)
- Missing graceful fallbacks when neuroimaging packages unavailable
- Dataset download/path issues
- Type hint errors
- Factory pattern implementation (`create_dataset_loader`)

**Test with:**
```python
from connectome_analysis.data.loaders import ABIDELoader, ADHD200Loader
loader = ABIDELoader()
# Check if this works without crashing
```

#### C. Debug `src/connectome_analysis/data/preprocessing.py`
**Common issues to check:**
- fMRI preprocessing pipeline errors
- Atlas loading failures
- Quality control metric computation bugs
- Memory issues with large neuroimaging data
- ROI extraction errors

**Key functions to test:**
- `preprocess_fmri()`
- `extract_roi_time_series()`
- `compute_motion_metrics()`
- `load_standard_atlases()`

#### D. Debug `src/connectome_analysis/data/connectome.py`
**Look for:**
- Connectivity matrix construction errors
- Graph theory metrics computation issues
- NetworkX version compatibility problems (especially community detection)
- Multi-modal connectivity combination bugs
- Memory efficiency issues with large matrices

**Test critical functions:**
- `build_functional_connectome()`
- `compute_graph_metrics()`
- `combine_modalities()`

### Phase 3: Model Architecture Debugging (Phase 4 Focus)
**If model files exist in `src/connectome_analysis/models/`:**

#### A. Baseline Models
- SVM, Random Forest, Logistic Regression implementations
- Feature vector extraction from connectomes
- Cross-validation setup

#### B. Graph Neural Networks
- GCN (Graph Convolutional Network) implementation
- GAT (Graph Attention Network) implementation
- PyTorch Geometric integration
- Graph data preparation and batching

### Phase 4: Configuration and Setup Debugging

#### A. Check `setup.py` or `pyproject.toml`
- Package metadata
- Dependency specifications
- Entry points
- Version constraints

#### B. Check `requirements.txt` and `environment.yml`
- Version conflicts
- Missing dependencies
- Optional dependency handling

#### C. Test package installation
```bash
pip install -e .  # Should work without errors
```

### Phase 5: Error Pattern Recognition

#### Common CGN Project Error Patterns:

1. **Optional Dependency Errors:**
   ```python
   # Bad:
   import nilearn
   
   # Good:
   try:
       import nilearn
       HAS_NILEARN = True
   except ImportError:
       HAS_NILEARN = False
   ```

2. **NetworkX Version Issues:**
   ```python
   # Check for deprecated community detection
   try:
       from networkx.algorithms.community import modularity
   except ImportError:
       from networkx.algorithms.community.quality import modularity
   ```

3. **PyTorch Geometric Compatibility:**
   ```python
   # Ensure proper graph data structure
   from torch_geometric.data import Data, Batch
   ```

4. **Memory Issues with Connectomes:**
   ```python
   # Large connectivity matrices need careful handling
   # Check for efficient sparse matrix usage
   ```

## Debugging Execution Plan

### Step 1: Run Diagnostics
```bash
# In project root
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "
try:
    import nilearn; print(f'Nilearn: {nilearn.__version__}')
except: print('Nilearn: Not available')
"
```

### Step 2: Test Core Imports
```python
# Create test_imports.py
import sys
import traceback

modules_to_test = [
    'connectome_analysis',
    'connectome_analysis.data',
    'connectome_analysis.data.loaders',
    'connectome_analysis.data.preprocessing', 
    'connectome_analysis.data.connectome'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")
        traceback.print_exc()
```

### Step 3: Systematic File Review
Go through each `.py` file and check for:
- Syntax errors
- Import errors
- Type hint issues
- Logic errors
- Performance bottlenecks

### Step 4: Run Tests
```bash
# If tests exist
python -m pytest tests/ -v
# Or run specific test files
python -m pytest tests/test_loaders.py -v
```

### Step 5: Create Minimal Working Examples
For each major component, create a minimal test:

```python
# test_minimal_loader.py
from connectome_analysis.data.loaders import create_dataset_loader

try:
    loader = create_dataset_loader('abide')
    print("✓ Loader creation successful")
except Exception as e:
    print(f"✗ Loader error: {e}")
```

## Expected Issues and Solutions

### 1. Missing Optional Dependencies
**Error:** `ImportError: No module named 'nilearn'`
**Solution:** Implement graceful fallbacks and clear error messages

### 2. NetworkX Community Detection
**Error:** `AttributeError: module 'networkx' has no attribute 'algorithms'`
**Solution:** Update to NetworkX 3.0+ compatible methods

### 3. PyTorch Geometric Version Conflicts
**Error:** Graph data structure incompatibilities
**Solution:** Ensure PyG ≥2.3.0 and compatible PyTorch version

### 4. Memory Issues
**Error:** `RuntimeError: CUDA out of memory` or system memory issues
**Solution:** Implement batch processing and memory-efficient operations

### 5. Path and Data Issues
**Error:** Dataset download or file path problems
**Solution:** Robust path handling and clear data directory structure

## Output Format
For each file debugged, provide:
1. **File:** `src/connectome_analysis/data/loaders.py`
2. **Status:** ✓ Working / ✗ Has Issues
3. **Issues Found:** List specific problems
4. **Fixes Applied:** Describe solutions
5. **Remaining Tasks:** What still needs attention

## Success Criteria
- All core modules import without errors
- Basic functionality works (data loading, preprocessing, connectome construction)
- No critical type checking errors
- Graceful handling of optional dependencies
- Memory-efficient operations for typical neuroscience datasets
- Ready for Phase 4 model implementation

Start with Phase 1 (environment) and work systematically through each phase. Focus on getting the core data pipeline working before moving to advanced features.