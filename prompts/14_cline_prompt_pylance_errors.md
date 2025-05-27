# Cline Task: Resolve Pylance Type Checking Errors in CGN Project

## Context
You are working on the **CGN (Connectome Graph Networks)** project - an AI-augmented connectome analysis system for neuroscience research. The project is in Phase 4 (Baseline Models and Initial GNNs) and has 4 persistent Pylance type checking errors that need to be resolved.

## Project Structure
```
CGN/
├── src/connectome_analysis/
│   ├── data/                    # Data handling (COMPLETE)
│   ├── models/                  # Model architectures
│   ├── training/                # Training pipelines
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utility functions
```

## Specific Pylance Errors to Fix

### 1. `src/connectome_analysis/evaluation/metrics.py` - Line 778
**Error:** `"stat_test_results" is possibly unbound`
**Issue:** Variable might be accessed before assignment in conditional blocks
**Required Fix:** Initialize `stat_test_results` properly to ensure it's always bound before use

### 2. `src/connectome_analysis/models/baseline.py` - Line 101  
**Error:** `Type "ndarray[Unknown, Unknown] | tuple[ndarray[Unknown, Unknown], ndarray[Unknown, Unknown]]" is not assignable to return type "ndarray[Unknown, Unknown]"`
**Issue:** `predict_proba` method can return either `np.ndarray` or `tuple` but is type-hinted to return only `np.ndarray`
**Required Fix:** Handle both return types properly and update type hints

### 3. `src/connectome_analysis/training/experiment.py` - Line 103
**Error:** `"model_type" is possibly unbound`  
**Issue:** Variable might be used before assignment
**Required Fix:** Ensure `model_type` is properly initialized before use

### 4. `src/connectome_analysis/training/trainer.py` - Line 329
**Error:** `Type "ndarray[Unknown, Unknown] | tuple[ndarray[Unknown, Unknown], ndarray[Unknown, Unknown]]" is not assignable to declared type "ndarray[Unknown, Unknown]"`
**Issue:** Same as baseline.py - `predict_proba` type handling
**Required Fix:** Handle both return types from `predict_proba` and cast appropriately

## Technical Requirements

### For Variable Initialization Errors (1 & 3):
- Initialize variables with appropriate default values at function start
- Use proper type hints: `variable_name: Type = default_value`
- Ensure variables are assigned before any conditional logic that might skip assignment
- Consider using `Optional[Type]` if variables can legitimately be `None`

### For predict_proba Type Errors (2 & 4):
- Import required types: `from typing import Union, Tuple, cast`
- Handle both `np.ndarray` and `tuple` return types from scikit-learn's `predict_proba`
- Use isinstance checks: `if isinstance(result, tuple): ...`
- Apply appropriate casting: `cast(np.ndarray, result)` when needed
- Update function return type hints to reflect actual possibilities

### Code Quality Standards:
- Maintain existing functionality - don't break working code
- Add comprehensive docstrings explaining type handling
- Use meaningful variable names
- Include error handling for edge cases
- Follow existing code style and patterns

## Neuroimaging Context
- This is scientific research code for brain connectivity analysis
- Functions handle large connectivity matrices and time series data
- Statistical tests are performed on neuroimaging results
- Machine learning models classify brain disorders (autism, ADHD)
- Code must be robust for research reproducibility

## Implementation Strategy

1. **Examine each error location** in the specified files and lines
2. **Identify the root cause** of each type checking issue
3. **Implement targeted fixes** that resolve the Pylance errors without changing functionality
4. **Test that fixes don't break existing code** by ensuring method signatures remain compatible
5. **Add appropriate type hints and imports** as needed
6. **Document any complex type handling** with inline comments

## Success Criteria
- [ ] All 4 Pylance errors resolved
- [ ] No new type checking errors introduced
- [ ] Existing functionality preserved
- [ ] Code follows project's type annotation standards
- [ ] Changes are minimal and focused only on type issues

## Additional Notes
- The project uses Python 3.8-3.10 with modern type hints
- Scientific computing packages (NumPy, SciPy, Scikit-learn) are core dependencies
- Some errors may be false positives from static analysis limitations
- Focus on runtime correctness while satisfying static type checking
- Use `# type: ignore` comments only as a last resort with explanation

## Files to Modify
1. `src/connectome_analysis/evaluation/metrics.py` (around line 778)
2. `src/connectome_analysis/models/baseline.py` (around line 101)
3. `src/connectome_analysis/training/experiment.py` (around line 103)  
4. `src/connectome_analysis/training/trainer.py` (around line 329)

Please examine each file, understand the context of the errors, and implement clean, targeted fixes that resolve the Pylance type checking issues while maintaining the scientific computing functionality of the CGN project.