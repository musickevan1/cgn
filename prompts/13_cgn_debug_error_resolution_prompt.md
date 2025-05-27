# 13_cgn_debug_error_resolution_prompt.md

# CGN Phase 4 Debug and Error Resolution Prompt

## Context and Current State

### Project Status
- **Phase**: Phase 4 Implementation (Baseline Models + Initial GNNs)
- **Recent Work**: Foundational architecture completed with new modules:
  - `models/baseline.py` (Enhanced with ensembles and feature selection)
  - `training/cross_validation.py` (Neuroimaging-specific CV strategies)
  - `models/gnn_models.py` (GCN, GAT, ConnectomeClassifier classes)
  - `models/graph_utils.py` (PyTorch Geometric integration)
  - `training/experiment.py` (Experiment orchestration)

### Known Technical Issues
1. **File Modification Problems**: Persistent issues with file corruption during modifications
2. **Pylance Type Errors**: Complex type hinting issues across multiple modules
3. **Library Integration Challenges**: sklearn, PyTorch, and PyTorch Geometric compatibility

### Current Error Status
**24 Pylance errors identified** across 6 files:
- `scripts/integration/prepare_abide_data.py` (2 errors - nibabel import)
- `scripts/validation/minimal_pipeline_test.py` (8 errors - type mismatches) 
- `src/connectome_analysis/evaluation/metrics.py` (6 errors - unbound variables)
- `src/connectome_analysis/models/baseline.py` (2 errors - redeclaration, return type)
- `src/connectome_analysis/training/cross_validation.py` (1 error - None assignment)
- `src/connectome_analysis/training/trainer.py` (1 error - assignment type)

## Debugging Methodology

### Phase 1: Error Identification and Categorization

#### Step 1: Collect All Current Errors
```bash
# Run comprehensive error detection
echo "=== PYTHON SYNTAX ERRORS ==="
find src/ -name "*.py" -exec python -m py_compile {} \; 2>&1

echo "=== IMPORT ERRORS ==="
python -c "
try:
    from src.connectome_analysis.models import baseline, gnn_models
    from src.connectome_analysis.training import cross_validation, experiment
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo "=== PYLANCE TYPE ERRORS ==="
# Run pylance/mypy if available
python -m mypy src/ --ignore-missing-imports 2>&1 || echo "Mypy not available"
```

#### Step 2: Categorize Errors by Type
- **Syntax Errors**: Immediate blocking issues
- **Import Errors**: Module resolution problems  
- **Type Errors**: Static analysis warnings
- **Runtime Errors**: Execution-time failures

### Phase 2: Systematic Error Resolution

#### Priority Order (Fix in this sequence)
1. **Syntax Errors** (Highest Priority - Blocks everything)
2. **Import Errors** (High Priority - Prevents testing)
3. **Runtime Errors** (Medium Priority - Breaks functionality)
4. **Type Errors** (Lower Priority - Static analysis only)

## Specific Error Fixes (Priority Order)

### 🔥 Critical Priority: Function Redeclaration
**File:** `src/connectome_analysis/models/baseline.py`
**Error:** Function `flatten_connectivity_matrix` declared twice (lines 16 and 207)
**Impact:** This causes function shadowing and unpredictable behavior

**Fix:**
```python
# Remove duplicate function definition
# Keep only one version - likely the more complete one at line 207
# Search for both occurrences and merge functionality if needed

# Step 1: Examine both function implementations
# Step 2: Merge any unique functionality 
# Step 3: Remove the duplicate at line 16
# Step 4: Update any references to use the remaining function
```

### 🚨 High Priority: Import and Unbound Variable Errors

#### 1. Nibabel Import Issues (prepare_abide_data.py)
**Lines 49, 60:** `"load" is not exported from module "nibabel"`

**Fix:**
```python
# Change from:
from nibabel import load

# To:
import nibabel as nib

# Update usage from:
img = load(file_path)

# To:
img = nib.load(file_path)
```

#### 2. Unbound Stats Variable (metrics.py)
**Lines 80, 98, 181:** `"stats" is unbound`

**Fix:**
```python
# Add at the top of metrics.py:
import scipy.stats as stats

# Or if already imported conditionally, ensure it's always available:
try:
    import scipy.stats as stats
except ImportError:
    stats = None
    
# Then add guards around usage:
if stats is not None:
    result = stats.wilcoxon(data1, data2)
else:
    raise ImportError("scipy.stats required for statistical tests")
```

### 🔧 Medium Priority: Type Mismatch Errors

#### 3. PyTorch Geometric Data Constructor Issues (minimal_pipeline_test.py)
**Lines 253, 266:** Type mismatches in Data object creation

**Fix:**
```python
# For lines 253 and 266, fix the PyTorch Geometric Data construction:

import torch
import numpy as np
from torch_geometric.data import Data

# Original problematic code around line 253:
# data = Data(data=connectivity_matrices.tolist(), labels=y_encoded.tolist())

# Fixed version:
def create_graph_data(connectivity_matrix: np.ndarray, label: int) -> Data:
    """Convert connectivity matrix to PyTorch Geometric Data object"""
    
    # Ensure we have a 2D matrix
    if connectivity_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {connectivity_matrix.ndim}D")
    
    n_nodes = connectivity_matrix.shape[0]
    
    # Create edge indices and edge attributes from connectivity matrix
    # Use upper triangle to avoid duplicate edges in undirected graph
    edge_indices = np.triu_indices(n_nodes, k=1)
    edge_weights = connectivity_matrix[edge_indices]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create node features (can be simple ones for now)
    x = torch.ones((n_nodes, 1), dtype=torch.float)
    
    # Create the Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label, dtype=torch.long)
    )
    
    return data

# Usage in the main code:
data_list = []
for i, (conn_matrix, label) in enumerate(zip(connectivity_matrices, y_encoded)):
    if isinstance(conn_matrix, np.ndarray):
        graph_data = create_graph_data(conn_matrix, label)
        data_list.append(graph_data)
    else:
        # Handle list case by converting to numpy first
        conn_array = np.array(conn_matrix)
        graph_data = create_graph_data(conn_array, label)
        data_list.append(graph_data)
```

#### 4. Dictionary Access on String (minimal_pipeline_test.py)
**Line 405:** Attempting `.get()` on string instead of dict

**Fix:**
```python
# Original problematic code:
# phenotypic.get('SITE_ID', 'Unknown')

# Add type checking:
def safe_phenotypic_access(phenotypic_data, key: str, default='Unknown'):
    """Safely access phenotypic data regardless of type"""
    if isinstance(phenotypic_data, dict):
        return phenotypic_data.get(key, default)
    elif isinstance(phenotypic_data, str):
        # Log warning and return default
        print(f"Warning: Expected dict for phenotypic data, got string: {phenotypic_data}")
        return default
    else:
        print(f"Warning: Unexpected phenotypic data type: {type(phenotypic_data)}")
        return default

# Usage:
site_id = safe_phenotypic_access(phenotypic, 'SITE_ID', 'Unknown')
dx_group = safe_phenotypic_access(phenotypic, 'DX_GROUP', 'Unknown') 
age = safe_phenotypic_access(phenotypic, 'AGE_AT_SCAN', 'Unknown')
```

### 🛠️ Lower Priority: Return Type and Assignment Issues

#### 5. Return Type Mismatch (baseline.py, trainer.py)
**baseline.py line 119, trainer.py line 324:** Tuple/array type conflicts

**Fix for baseline.py:**
```python
# Update the predict_proba method to handle tuple returns:
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities."""
    if hasattr(self.model, 'predict_proba'):
        result = self.model.predict_proba(X)
        # Handle case where sklearn returns tuple (rare, but possible)
        if isinstance(result, tuple):
            return result[0]  # Return probabilities, not decision function
        return result
    else:
        # Fallback for models without predict_proba
        predictions = self.model.predict(X)
        n_classes = len(np.unique(predictions))
        proba = np.zeros((len(predictions), n_classes))
        for i, pred in enumerate(predictions):
            proba[i, int(pred)] = 1.0
        return proba
```

**Fix for trainer.py:**
```python
# Around line 324, handle the assignment type issue:
predictions = self.model.predict_proba(X_test)
if isinstance(predictions, tuple):
    y_pred: np.ndarray = predictions[0]
else:
    y_pred: np.ndarray = predictions
```

#### 6. None Assignment Issue (cross_validation.py)
**Line 27:** `Expression of type "None" cannot be assigned to parameter of type "int"`

**Fix:**
```python
# Look at line 27 and fix the None assignment:
# Likely something like: some_parameter=None where int expected

# Example fix:
# Original: random_state=None
# Fixed: random_state=42  # or use a proper default

# Or with proper handling:
def some_function(random_state: Optional[int] = None):
    if random_state is None:
        random_state = 42  # Default value
    # Use random_state as int from here
```

#### 7. Tensor Callable Issue (metrics.py)
**Line 270:** `Object of type "Tensor" is not callable`

**Fix:**
```python
# Look at line 270 for tensor being called as function
# Common mistake: tensor_name() instead of tensor_name.method()

# Possible fixes:
# If trying to get tensor value:
# result = tensor_var()  # Wrong
result = tensor_var.item()  # Correct for scalar
# or
result = tensor_var.numpy()  # Correct for array

# If trying to call a method:
# result = tensor_var()  # Wrong  
result = tensor_var.method_name()  # Correct

# If tensor contains a function (rare):
# result = tensor_var()  # Wrong
result = tensor_var.item()()  # If tensor contains callable
```

#### 8. Possibly Unbound Variables
**metrics.py line 763, experiment.py line 103:** Variables may not be initialized

**Fix:**
```python
# For metrics.py line 763 (stat_test_results):
stat_test_results = None  # Initialize at function start

# Then ensure all code paths assign a value:
if some_condition:
    stat_test_results = perform_test()
else:
    stat_test_results = {'default': 'no_test'}

# For experiment.py line 103 (model_type):
# Ensure model_type is always defined before use
model_type = 'default'  # Set default value

# Or restructure to ensure assignment:
for config in model_configs:
    model_type = config.get('type', 'svm')  # Always has a value
    # ... use model_type
```

## Specific Debugging Scripts

### Script 1: Comprehensive Error Checker
```python
# Create: scripts/debug/error_checker.py
"""
Comprehensive error detection for CGN Phase 4
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_syntax_errors():
    """Check all Python files for syntax errors"""
    errors = []
    src_path = Path("src")
    
    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            errors.append({
                'file': str(py_file),
                'line': e.lineno,
                'error': str(e),
                'type': 'syntax'
            })
    
    return errors

def check_import_errors():
    """Check module imports"""
    modules_to_test = [
        'src.connectome_analysis.models.baseline',
        'src.connectome_analysis.models.gnn_models',
        'src.connectome_analysis.training.cross_validation',
        'src.connectome_analysis.training.experiment'
    ]
    
    errors = []
    for module in modules_to_test:
        try:
            spec = importlib.util.spec_from_file_location(
                module, module.replace('.', '/') + '.py'
            )
            imported_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported_module)
        except Exception as e:
            errors.append({
                'module': module,
                'error': str(e),
                'type': 'import'
            })
    
    return errors

def main():
    print("🔍 CGN Phase 4 Error Detection")
    print("=" * 50)
    
    # Check syntax errors
    syntax_errors = check_syntax_errors()
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors:
            print(f"  📁 {error['file']}:{error['line']} - {error['error']}")
    else:
        print("✅ No syntax errors found")
    
    # Check import errors
    import_errors = check_import_errors()
    if import_errors:
        print(f"❌ Found {len(import_errors)} import errors:")
        for error in import_errors:
            print(f"  📦 {error['module']} - {error['error']}")
    else:
        print("✅ No import errors found")
    
    return len(syntax_errors) + len(import_errors)

if __name__ == "__main__":
    error_count = main()
    sys.exit(error_count)
```

### Script 2: Incremental Test Framework
```python
# Create: scripts/debug/incremental_test.py
"""
Test CGN components incrementally to isolate issues
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

def test_basic_imports():
    """Test if basic modules can be imported"""
    tests = []
    
    try:
        from src.connectome_analysis.data.loaders import ABIDELoader
        tests.append(("ABIDELoader", "✅"))
    except Exception as e:
        tests.append(("ABIDELoader", f"❌ {e}"))
    
    try:
        from src.connectome_analysis.models.baseline import EnhancedBaselineClassifier
        tests.append(("EnhancedBaselineClassifier", "✅"))
    except Exception as e:
        tests.append(("EnhancedBaselineClassifier", f"❌ {e}"))
    
    try:
        from src.connectome_analysis.training.cross_validation import leave_one_site_out_cv
        tests.append(("leave_one_site_out_cv", "✅"))
    except Exception as e:
        tests.append(("leave_one_site_out_cv", f"❌ {e}"))
    
    return tests

def test_synthetic_data():
    """Test with synthetic data"""
    tests = []
    
    try:
        # Create synthetic data
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)
        sites = np.random.randint(0, 3, 50)
        
        # Test cross-validation
        from src.connectome_analysis.training.cross_validation import leave_one_site_out_cv
        splits = leave_one_site_out_cv(X, y, sites)
        tests.append(("Cross-validation", f"✅ Generated {len(splits)} splits"))
        
    except Exception as e:
        tests.append(("Cross-validation", f"❌ {e}"))
    
    return tests

def main():
    print("🧪 CGN Incremental Testing")
    print("=" * 40)
    
    # Test imports
    print("\n📦 Testing Imports:")
    import_tests = test_basic_imports()
    for test_name, result in import_tests:
        print(f"  {test_name}: {result}")
    
    # Test with synthetic data
    print("\n🔬 Testing with Synthetic Data:")
    synthetic_tests = test_synthetic_data()
    for test_name, result in synthetic_tests:
        print(f"  {test_name}: {result}")

if __name__ == "__main__":
    main()
```

## Error Resolution Workflow

### Step-by-Step Process

## Step-by-Step Resolution Plan

### Phase 1: Critical Fixes (Address Immediately)
```bash
# 1. Fix function redeclaration in baseline.py
# Search for duplicate flatten_connectivity_matrix functions
grep -n "def flatten_connectivity_matrix" src/connectome_analysis/models/baseline.py

# 2. Fix nibabel imports
sed -i 's/from nibabel import load/import nibabel as nib/g' scripts/integration/prepare_abide_data.py
sed -i 's/load(/nib.load(/g' scripts/integration/prepare_abide_data.py

# 3. Add scipy.stats import to metrics.py
echo "import scipy.stats as stats" >> temp_import.py
cat src/connectome_analysis/evaluation/metrics.py >> temp_import.py
mv temp_import.py src/connectome_analysis/evaluation/metrics.py
```

### Phase 2: Type Safety Fixes (Medium Priority)  
```python
# Add to top of each problematic file:

# For minimal_pipeline_test.py:
from typing import Union, List, Any, Optional
import numpy as np
import torch
from torch_geometric.data import Data

# For cross_validation.py:
from typing import Optional
# Fix None parameter by providing default: random_state: Optional[int] = 42

# For experiment.py:
# Initialize model_type at function start: model_type = 'svm'
```

### Phase 3: Validation Tests
```python
# Create quick validation script:
# scripts/debug/validate_fixes.py

import sys
import importlib.util

def test_imports():
    """Test all fixed imports"""
    tests = [
        ('nibabel as nib', 'import nibabel as nib'),
        ('scipy.stats', 'import scipy.stats as stats'),
        ('baseline module', 'from src.connectome_analysis.models import baseline'),
        ('metrics module', 'from src.connectome_analysis.evaluation import metrics')
    ]
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: OK")
        except Exception as e:
            print(f"❌ {name}: {e}")

def test_function_definitions():
    """Check for duplicate function definitions"""
    import ast
    from pathlib import Path
    
    baseline_file = Path("src/connectome_analysis/models/baseline.py")
    if baseline_file.exists():
        with open(baseline_file) as f:
            tree = ast.parse(f.read())
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        duplicates = [f for f in functions if functions.count(f) > 1]
        
        if duplicates:
            print(f"❌ Duplicate functions found: {duplicates}")
        else:
            print("✅ No duplicate functions")

if __name__ == "__main__":
    test_imports()
    test_function_definitions()
```

## File-by-File Fix Instructions

### 1. `scripts/integration/prepare_abide_data.py` (2 errors)
```bash
# Lines 49, 60: nibabel import issues
# Quick fix:
sed -i 's/from nibabel import load/import nibabel as nib/' scripts/integration/prepare_abide_data.py
sed -i 's/load(/nib.load(/g' scripts/integration/prepare_abide_data.py

# Verification:
python -c "import nibabel as nib; print('nibabel import OK')"
```

### 2. `src/connectome_analysis/models/baseline.py` (2 errors)
```python
# Critical: Function redeclaration at lines 16 and 207
# Step 1: Examine both functions
grep -A 10 -B 2 "def flatten_connectivity_matrix" src/connectome_analysis/models/baseline.py

# Step 2: Remove duplicate (likely line 16 version)
# Step 3: Fix return type issue at line 119
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    result = self.model.predict_proba(X)
    return result[0] if isinstance(result, tuple) else result
```

### 3. `src/connectome_analysis/evaluation/metrics.py` (6 errors)
```python
# Add at very top of file:
import scipy.stats as stats

# Fix line 270 tensor callable issue:
# Change: result = tensor_var()
# To: result = tensor_var.item() or tensor_var.numpy()

# Fix line 763 unbound variable:
# Initialize at function start:
stat_test_results = None
```

### 4. `scripts/validation/minimal_pipeline_test.py` (8 errors)
```python
# Major rewrite needed for PyTorch Geometric compatibility
# Add helper function at top:

def create_brain_graph(connectivity_matrix, label):
    """Convert connectivity matrix to PyTorch Geometric Data"""
    import torch
    from torch_geometric.data import Data
    
    # Handle different input types
    if not isinstance(connectivity_matrix, np.ndarray):
        connectivity_matrix = np.array(connectivity_matrix)
    
    n_nodes = connectivity_matrix.shape[0]
    
    # Create edges from upper triangle
    edge_indices = np.triu_indices(n_nodes, k=1)
    edge_weights = connectivity_matrix[edge_indices]
    
    data = Data(
        x=torch.ones((n_nodes, 1), dtype=torch.float),
        edge_index=torch.tensor(np.vstack(edge_indices), dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float),
        y=torch.tensor(int(label), dtype=torch.long)
    )
    return data

# Replace lines 253, 266 with:
data_list = [create_brain_graph(conn, label) 
            for conn, label in zip(connectivity_matrices, y_encoded)]

# Fix line 405 dictionary access:
def safe_get(data, key, default='Unknown'):
    return data.get(key, default) if isinstance(data, dict) else default

site_id = safe_get(phenotypic, 'SITE_ID')
```

### 5. `src/connectome_analysis/training/cross_validation.py` (1 error)
```python
# Line 27: None assignment to int parameter
# Change: random_state=None
# To: random_state: Optional[int] = 42

# Add import:
from typing import Optional
```

### 6. `src/connectome_analysis/training/trainer.py` (1 error)
```python
# Line 324: Assignment type mismatch
# Change:
y_pred: np.ndarray = self.model.predict_proba(X_test)

# To:
predictions = self.model.predict_proba(X_test)
y_pred: np.ndarray = predictions[0] if isinstance(predictions, tuple) else predictions
```

### 7. `src/connectome_analysis/training/experiment.py` (1 error)
```python
# Line 103: Possibly unbound model_type
# Add at function start:
model_type = 'svm'  # Default value

# Or ensure assignment before use:
for config in configs:
    model_type = config.get('type', 'svm')
    # ... rest of loop
```

## Success Criteria

### Minimum Working State
- [ ] All Python files compile without syntax errors
- [ ] Core modules can be imported successfully
- [ ] Basic functionality works with synthetic data
- [ ] Cross-validation runs without crashing

### Optimal State
- [ ] All major Pylance warnings resolved
- [ ] End-to-end pipeline runs on real data
- [ ] Performance meets baseline benchmarks
- [ ] Code passes basic unit tests

## Documentation Requirements

### For Each Fixed Error
- Document the error type and cause
- Explain the resolution approach
- Note any trade-offs or limitations
- Update code comments as needed

### Overall Debug Session
- Summarize total errors found and fixed
- Identify recurring patterns for future prevention
- Document any workarounds or temporary solutions
- Create guidance for similar issues

---

**Instructions for Use**: 
1. Provide the specific error messages and context
2. Run the diagnostic scripts to assess current state
3. Follow the systematic resolution workflow
4. Focus on syntax and import errors first
5. Document all fixes for future reference

**Note**: This prompt assumes specific error details will be provided. Update the error-specific sections based on the actual errors encountered in your CGN Phase 4 implementation.
