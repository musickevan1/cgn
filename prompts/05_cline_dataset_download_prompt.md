# CGN Dataset Download and Import Task

## Objective
Download and import the ABIDE and ADHD-200 neuroimaging datasets into the CGN project structure, ensuring proper data organization and validation.

## Context
The CGN (Connectome Graph Networks) project is at Phase 4 and needs the core neuroimaging datasets to begin implementing baseline models and GNNs. The project already has data loading infrastructure in place (`src/connectome_analysis/data/loaders.py`) that needs to be utilized.

## Current Project Structure
```
CGN/
├── src/connectome_analysis/
│   └── data/
│       ├── loaders.py           # Dataset loading classes (implemented)
│       ├── preprocessing.py      # Preprocessing pipeline (implemented)
│       └── connectome.py        # Connectome construction (implemented)
├── data/
│   ├── raw/                     # Raw downloaded datasets (target location)
│   ├── processed/               # Processed connectomes (future use)
│   └── external/                # External atlases/templates
├── scripts/
│   └── data/
│       └── download_datasets.py # Download script (needs implementation)
└── configs/                     # Configuration files
```

## Tasks to Complete

### 1. Implement Dataset Download Script
Create `scripts/data/download_datasets.py` that:
- Uses the existing `ABIDELoader` and `ADHD200Loader` classes from `src/connectome_analysis/data/loaders.py`
- Downloads both ABIDE and ADHD-200 datasets via nilearn
- Organizes data in the `data/raw/` directory structure
- Includes proper error handling and progress reporting
- Supports command-line arguments for dataset selection

**Expected CLI usage:**
```bash
python scripts/data/download_datasets.py --dataset abide
python scripts/data/download_datasets.py --dataset adhd200
python scripts/data/download_datasets.py --dataset all
```

### 2. Verify Data Loading Infrastructure
Test and validate the existing data loading classes:
- Check `ABIDELoader` and `ADHD200Loader` functionality
- Ensure graceful handling of missing neuroimaging packages
- Verify phenotypic data loading works correctly
- Test the factory function `create_dataset_loader()`

### 3. Create Data Validation Script
Implement a validation script that:
- Verifies downloaded data integrity
- Checks file formats and expected structure
- Reports dataset statistics (number of subjects, sites, etc.)
- Validates phenotypic data completeness

### 4. Update Configuration
Create or update configuration files for:
- Dataset paths and URLs
- Download settings and timeouts
- Data organization preferences
- Cache and temporary file locations

## Technical Requirements

### Dependencies
The project uses these key packages for data handling:
```python
import nilearn
import nibabel
import pandas as pd
import numpy as np
from pathlib import Path
```

### Error Handling
- Graceful degradation when neuroimaging packages unavailable
- Network timeout and retry logic for downloads
- Disk space checking before downloads
- Clear error messages for troubleshooting

### Data Organization
Expected directory structure after download:
```
data/
├── raw/
│   ├── abide/
│   │   ├── neuroimaging/        # fMRI NIfTI files
│   │   └── phenotypic/          # Subject demographics/clinical data
│   └── adhd200/
│       ├── neuroimaging/        # fMRI NIfTI files
│       └── phenotypic/          # Subject demographics/clinical data
└── external/
    └── atlases/                 # Brain parcellation atlases
```

### Performance Considerations
- Large datasets (multi-GB downloads)
- Progress bars for user feedback
- Resume capability for interrupted downloads
- Parallel downloads where appropriate

## Expected Outputs

### 1. Download Script
A robust `scripts/data/download_datasets.py` that:
- Integrates with existing CGN data loading infrastructure
- Provides clear progress feedback
- Handles common download issues
- Logs activities for debugging

### 2. Validation Report
After successful download, generate a report showing:
- Dataset summary statistics
- File integrity checks
- Phenotypic data completeness
- Any issues or warnings

### 3. Updated Documentation
Update project documentation with:
- Dataset download instructions
- Troubleshooting guide for common issues
- Data organization explanation
- Next steps for preprocessing

## Quality Checks
Before marking this task complete, verify:
- [ ] Both ABIDE and ADHD-200 datasets successfully downloaded
- [ ] Data organized in expected directory structure
- [ ] Existing `loaders.py` classes can successfully load the data
- [ ] Phenotypic data accessible and properly formatted
- [ ] Error handling works for various failure scenarios
- [ ] Documentation updated with usage instructions

## Notes for Implementation
- The project already has robust data loading infrastructure - leverage it
- Focus on integration with existing code rather than reimplementation
- Consider memory usage for large neuroimaging datasets
- Ensure compatibility with the project's Docker containerization
- Follow the project's coding standards (type hints, docstrings, error handling)

## Success Criteria
- Datasets are fully downloaded and validated
- Data loading pipeline works end-to-end
- Clear documentation for reproduction
- Foundation ready for Phase 4 model implementation

This task establishes the critical data foundation needed for the CGN project's next development phase.