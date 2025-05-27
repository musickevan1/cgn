# Data Handling

This document provides instructions for downloading, organizing, and validating the neuroimaging datasets used in the CGN project.

## Dataset Download

The CGN project utilizes the ABIDE and ADHD-200 datasets. These can be downloaded using the provided Python script.

To download a specific dataset, navigate to the project's root directory in your terminal and run:

```bash
python scripts/data/download_datasets.py --dataset [dataset_name]
```

Replace `[dataset_name]` with either `abide` or `adhd200`.

To download both datasets, use the `all` option:

```bash
python scripts/data/download_datasets.py --dataset all
```

The script will download the data using nilearn and organize it in the `data/raw/` directory.

**Note:** The initial download may take a significant amount of time and disk space as these are large neuroimaging datasets. The script currently downloads a subset of subjects for rapid development as configured in `configs/data/data_config.yaml`.

## Data Organization

Downloaded datasets are organized within the `data/raw/` directory with the following structure:

```
data/
└── raw/
    ├── abide/
    │   ├── neuroimaging/        # fMRI NIfTI files
    │   └── phenotypic/          # Subject demographics/clinical data
    └── adhd200/
        ├── neuroimaging/        # fMRI NIfTI files
        └── phenotypic/          # Subject demographics/clinical data
```

This structure ensures consistency and compatibility with the project's data loading infrastructure.

## Data Validation

After downloading, you can validate the integrity and structure of the datasets using the validation script:

```bash
python scripts/data/validate_datasets.py --dataset [dataset_name] --data_dir data/raw
```

Replace `[dataset_name]` with `abide`, `adhd200`, or `all`. The `--data_dir` argument should point to the base directory containing the raw data (default is `data/raw`).

The validation script checks for expected directories, attempts to load phenotypic data, and performs basic checks on neuroimaging files.

## Troubleshooting

- **Neuroimaging packages not found:** If you encounter errors related to `nibabel` or `nilearn`, ensure you have installed the necessary dependencies. Refer to the project's `requirements.txt` or `environment.yml` for the required packages.
- **Download interrupted:** The nilearn fetchers used by the download script have some resume capability. You can often rerun the download command to continue from where it left off.
- **Disk space:** Ensure you have sufficient disk space before attempting to download the datasets.
- **Network issues:** Check your internet connection if downloads fail. The script includes basic error handling for download failures.

## Next Steps: Preprocessing

Once the datasets are downloaded and validated, the next step is to run the preprocessing pipeline. Refer to the documentation on preprocessing for instructions on how to convert the raw neuroimaging data into processed connectomes.
