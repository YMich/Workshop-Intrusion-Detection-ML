# Data Directory

Large datasets are intentionally **not stored in the GitHub repository**.

They are distributed separately through the accompanying Google Drive package.

## Expected Local Structure

After downloading the data package, this directory should contain:

```text
data/
├── raw/
└── ingested/
    ├── dataset1_ingested.csv
    ├── dataset2_ingested.csv
    └── dataset3_ingested.csv
```

## Main Evaluation Data

The primary Dataset-2 / Dataset-3 experiments expect:

```text
data/ingested/dataset2_ingested.csv
data/ingested/dataset3_ingested.csv
```

Dataset 1 is used for appendix/generalization evaluation:

```text
data/ingested/dataset1_ingested.csv
```

## `raw/`

`data/raw/` is required only when reproducing the earlier raw-PCAP extraction/ingestion stages.

For evaluation of the already-ingested datasets and frozen models, the ingested CSV files are the important inputs.

## Installation

Copy the downloaded Google Drive `data/` contents into the repository's existing `data/` directory while preserving filenames and directory structure.

Do not rename the ingested CSV files unless the corresponding project paths are updated.
