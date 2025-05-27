# CGN Project - Pylance Error Resolution

## Task Overview
Fix remaining Pylance type checking errors in the CGN (Connectome Graph Networks) project. These are primarily type annotation issues, import corrections, and data type mismatches that need to be resolved for clean code quality.

## Priority Order (Most Critical First)

### 1. Import and Module Issues (CRITICAL)

**File:** `scripts/integration/prepare_abide_data.py` (Line 5)
- **Issue:** `"load" is not exported from module "nibabel"`
- **Fix:** Change `from nibabel import load` to `from nibabel.loadsave import load`
- **Alternative:** Use `import nibabel as nib` and call `nib.load()`

**File:** `src/connectome_analysis/evaluation/metrics.py` (Lines 80, 98, 181)
- **Issue:** `"stats" is unbound` - missing scipy.stats import
- **Fix:** Add `from scipy import stats` at the top of the file
- **Context:** Used for `stats.wilcoxon`, `stats.ttest_rel`, and `stats.t`

### 2. Data Type and Argument Issues (HIGH PRIORITY)

**File:** `src/connectome_analysis/data/preprocessing.py` (Line 66)
- **Issue:** `standardize` parameter expects `str` but receives `bool`
- **Fix:** Check nilearn's `signal.clean()` API - likely needs `standardize="zscore"` or `standardize=False` as string
- **Context:** This is in the preprocessing pipeline, critical for data quality

**File:** `src/connectome_analysis/models/baseline.py` (Line 52)
- **Issue:** Return type mismatch - function returns tuple but expects ndarray
- **Fix:** Handle the case where sklearn methods return both predictions and probabilities
- **Solution:** Extract only the prediction array, not the tuple

**File:** `src/connectome_analysis/training/trainer.py` (Lines 315, 318, 319)
- **Issue:** Type mismatches in metric computation and logging
- **Fix:** Add proper type casting for PyTorch tensors and numpy arrays
- **Context:** This affects training loop metrics and logging

### 3. Data Structure Issues (MEDIUM PRIORITY)

**File:** `scripts/validation/minimal_pipeline_test.py` (Lines 253, 266)
- **Issue:** Data type mismatches in mock data creation
- **Context:** These are in test/validation code, less critical but should be clean
- **Fix:** Ensure proper numpy array handling and list conversion

**File:** `scripts/validation/minimal_pipeline_test.py` (Line 405)
- **Issue:** Calling `.get()` on string instead of dictionary
- **Fix:** Check if the variable should be a dictionary or if the access pattern is wrong

### 4. Tensor/Array Handling Issues (MEDIUM PRIORITY)

**File:** `src/connectome_analysis/evaluation/metrics.py` (Line 270)
- **Issue:** Trying to call a Tensor as a function
- **Fix:** Check if this should be tensor indexing `tensor[...]` instead of `tensor(...)`

**File:** `src/connectome_analysis/evaluation/metrics.py` (Line 762)
- **Issue:** `stat_test_results` possibly unbound
- **Fix:** Initialize variable properly or add conditional checks

## Detailed Fix Instructions

### Step 1: Fix Critical Imports
```python
# In scripts/integration/prepare_abide_data.py
# Change line 5 from:
from nibabel import load
# To:
import nibabel as nib
# And update usage from load() to nib.load()

# In src/connectome_analysis/evaluation/metrics.py
# Add at top of file:
from scipy import stats
```

### Step 2: Fix Data Type Issues
```python
# In src/connectome_analysis/data/preprocessing.py
# Check nilearn documentation and fix standardize parameter
# Likely change from:
standardize=True
# To:
standardize="zscore"  # or "psc" or False as string
```

### Step 3: Fix Return Type Issues
```python
# In src/connectome_analysis/models/baseline.py
# Handle sklearn methods that might return tuples:
predictions = model.predict_proba(X)
if isinstance(predictions, tuple):
    return predictions[0]  # or predictions[1] depending on what's needed
return predictions
```

### Step 4: Fix PyTorch/NumPy Type Issues
```python
# In src/connectome_analysis/training/trainer.py
# Add proper type conversion:
if isinstance(y_pred, tuple):
    y_pred = y_pred[0]
y_pred = y_pred.detach().cpu().numpy() if hasattr(y_pred, 'detach') else y_pred

# For logging values, ensure they're scalars:
metric_value = float(metric_value.item()) if hasattr(metric_value, 'item') else float(metric_value)
```

## Testing After Fixes

1. **Run type checker:** `pylance` or `mypy src/`
2. **Test imports:** `python -c "from src.connectome_analysis.evaluation.metrics import *"`
3. **Run minimal test:** `python scripts/validation/minimal_pipeline_test.py`
4. **Verify data loading:** `python scripts/integration/prepare_abide_data.py --help`

## Context Notes

- This is a neuroscience research project using PyTorch, PyTorch Geometric, and neuroimaging libraries
- The code handles brain connectivity matrices (connectomes) and applies machine learning
- Type safety is important for research reproducibility
- The project is in active development phase, so fixes should maintain functionality

## Success Criteria

- [ ] All Pylance errors resolved
- [ ] No runtime import errors
- [ ] Minimal pipeline test runs successfully
- [ ] Type annotations are correct and meaningful
- [ ] Code maintains its scientific computing functionality

## Files to Modify (In Order)

1. `src/connectome_analysis/evaluation/metrics.py` - Add scipy.stats import
2. `scripts/integration/prepare_abide_data.py` - Fix nibabel import
3. `src/connectome_analysis/data/preprocessing.py` - Fix standardize parameter
4. `src/connectome_analysis/models/baseline.py` - Fix return type handling
5. `src/connectome_analysis/training/trainer.py` - Fix tensor/array type issues
6. `scripts/validation/minimal_pipeline_test.py` - Fix test data type issues

Focus on the critical imports first, then work through the data type issues systematically.