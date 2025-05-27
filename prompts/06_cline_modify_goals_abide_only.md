# CGN Project Scope Modification: Focus on ABIDE Dataset Only

## Objective
Modify the CGN project to focus exclusively on the ABIDE (Autism Brain Imaging Data Exchange) dataset, removing ADHD-200 support temporarily due to technical issues.

## Context
The CGN project was originally designed to support both ABIDE and ADHD-200 datasets. However, ADHD-200 is causing implementation issues that are blocking progress. To maintain development momentum and meet the 1-2 month rapid prototyping timeline, we're streamlining to focus on ABIDE only.

## Scope Changes Required

### 1. Update Project Documentation
Modify the following files to reflect ABIDE-only focus:

**Primary targets:**
- `README.md` - Update project description and examples
- `CGN_PROJECT_CONTEXT.md` - Revise objectives and dataset references
- Any existing documentation mentioning dual-dataset support

**Changes needed:**
- Remove references to "multiple neuroimaging datasets (ABIDE, ADHD-200)"
- Update to "ABIDE neuroimaging dataset for autism spectrum disorder research"
- Simplify examples to show ABIDE-only workflows
- Update target use cases to focus on ASD vs control classification

### 2. Refactor Data Loading Infrastructure
Modify `src/connectome_analysis/data/loaders.py`:

**Current state:** Contains both `ABIDELoader` and `ADHD200Loader` classes
**Target state:** Prioritize `ABIDELoader`, deprecate or remove `ADHD200Loader`

**Specific changes:**
- Keep `ABIDELoader` class fully functional and well-tested
- Comment out or remove `ADHD200Loader` class implementation
- Update `create_dataset_loader()` factory function to handle ABIDE only
- Modify `get_available_datasets()` to return only ABIDE options
- Update all docstrings and type hints accordingly

### 3. Simplify Download Scripts
Update `scripts/data/download_datasets.py`:

**Remove ADHD-200 support:**
- Remove `--dataset adhd200` command-line option
- Simplify to support only `--dataset abide` or default behavior
- Remove ADHD-200 specific error handling and validation
- Update help text and documentation

**Expected CLI usage after changes:**
```bash
python scripts/data/download_datasets.py              # Downloads ABIDE by default
python scripts/data/download_datasets.py --dataset abide  # Explicit ABIDE download
```

### 4. Update Directory Structure
Simplify the expected data organization:

**Before:**
```
data/
├── raw/
│   ├── abide/
│   └── adhd200/
```

**After:**
```
data/
├── raw/
│   └── abide/
│       ├── neuroimaging/
│       └── phenotypic/
```

### 5. Modify Configuration Files
Update any configuration files to remove ADHD-200 references:
- Remove ADHD-200 dataset URLs and paths
- Simplify dataset configuration to ABIDE-only
- Update default settings for single-dataset workflow

### 6. Revise Project Examples and Tests
Update code examples and test cases:
- Modify example notebooks to use ABIDE data only
- Update test cases in `tests/` directory
- Revise any documentation examples showing dataset selection

### 7. Update Phase 4 Implementation Plan
Adjust the Phase 4 baseline model implementation:

**Focus areas:**
- ASD vs Control binary classification using ABIDE
- Functional connectivity analysis for autism spectrum disorder
- Graph neural network architectures optimized for ASD detection

**Remove:**
- Multi-class ADHD subtype classification
- ADHD-specific preprocessing considerations
- Cross-dataset validation between ABIDE and ADHD-200

## Implementation Priority

### High Priority (Complete First)
1. **Documentation updates** - README, project context
2. **Data loader refactoring** - Make ABIDELoader primary focus
3. **Download script simplification** - Remove ADHD-200 options

### Medium Priority
4. **Directory structure cleanup** - Remove ADHD-200 placeholders
5. **Configuration updates** - Simplify to single dataset
6. **Test case modifications** - ABIDE-focused testing

### Low Priority (Future Cleanup)
7. **Example notebooks** - Update when created
8. **Advanced configuration** - Can be addressed in later phases

## Benefits of This Approach

### Reduced Complexity
- Simpler codebase with fewer edge cases
- Faster development and testing cycles
- Clearer documentation and examples

### Better Focus
- Deep expertise in ASD classification using functional connectivity
- Optimized preprocessing pipeline for ABIDE dataset characteristics
- More robust implementation for single use case

### Maintained Flexibility
- Code architecture supports adding datasets in future phases
- ADHD-200 can be re-added once technical issues resolved
- Foundation remains solid for multi-dataset expansion

## Quality Checklist
Before considering this modification complete:

- [ ] All documentation updated to reflect ABIDE-only scope
- [ ] Data loading infrastructure works seamlessly with ABIDE
- [ ] Download scripts function correctly for ABIDE dataset
- [ ] No broken references to ADHD-200 in codebase
- [ ] Test cases pass with modified scope
- [ ] Project still meets core Phase 4 objectives

## Future Considerations
- **Phase 6+**: Consider re-adding ADHD-200 support after core functionality proven
- **Alternative datasets**: Could explore other neuroimaging datasets if needed
- **Multi-modal expansion**: Focus on combining structural/functional data within ABIDE

## Success Criteria
- Clean, focused codebase supporting ABIDE dataset exclusively
- Faster development velocity for Phase 4 implementation
- Robust foundation for ASD classification using graph neural networks
- Clear path forward for baseline model implementation

This modification maintains the project's core scientific objectives while removing technical blockers that impede development progress.