# PCAP Extraction and CICFlowMeter Alignment

This directory contains the source code used to extract and align the network-flow CSV files from the project's raw PCAP/PCAPNG sources.

It is included in the final submission for **data provenance and reproducibility**. The reported machine-learning experiments do **not** require this stage to be rerun because the submission's executable ML pipeline begins from the finalized files under `data/ingested/`.

## Pipeline Position

```text
Raw PCAP / PCAPNG + metadata
            |
            v
      src/extraction/
            |
            v
       data/extracted/
            |
            v
       src/ingestion/
            |
            v
        data/ingested/
            |
            v
 preprocessing -> feature engineering -> models -> hybrid / LLM
```

`src/extraction/` produces aligned extraction outputs under `data/extracted/`. The separate `src/ingestion/` stage then assembles those outputs into the canonical `data/ingested/` datasets used by the final models.

There is no separate `data/combined/` stage in the final submission pipeline.

## Directory Contents

```text
src/extraction/
├── README.md
├── cicflowmeter_common.py
├── config_based_malicious_common.py
├── pipeline_config.py
├── prepare_cicids2017_friday_benign_https.py
├── run_all_extractions.py
├── run_benign_cicflowmeter_aligned.py
├── run_malicious_cicflowmeter_aligned.py
├── run_dataset2_hikari_benign_https_aligned_parallel.py
├── run_dataset2_malicious_config_aligned_parallel.py
├── run_dataset3_benign_ctu_https_aligned_parallel.py
├── run_dataset3_malicious_config_aligned_parallel.py
└── setup_project_folders.py
```

## Main Entry Point

To run all extraction jobs in their required order:

```bash
python src/extraction/run_all_extractions.py
```

Dataset 1 benign extraction runs first because it establishes the reference CICFlowMeter schema used to align the remaining extraction outputs.

The runner executes:

1. Dataset 1 benign PCAP extraction.
2. Dataset 1 malicious HTTPS extraction.
3. Dataset 2 HIKARI-backed benign HTTPS extraction.
4. Dataset 2 malicious configuration-aware extraction.
5. Dataset 3 CTU benign HTTPS extraction.
6. Dataset 3 malicious configuration-aware extraction.
7. CICIDS2017 Friday benign HTTPS preparation for Dataset 1.

The old `concatenate_extracted_datasets.py` stage is intentionally not part of the final runner. Extracted outputs are consumed by `src/ingestion/` instead.

## Project Paths

`pipeline_config.py` resolves the repository root relative to its own location:

```text
<PROJECT_ROOT>/src/extraction/pipeline_config.py
```

No machine-specific absolute project path is required.

## CICFlowMeter Docker Image

The original extraction environment used CICFlowMeter through Docker. The original local image ID is retained as the default for provenance:

```text
1453ede6f6e9
```

On another machine, set the `CICFLOWMETER_DOCKER_IMAGE` environment variable to an installed equivalent image name, tag, or image ID before running extraction.

PowerShell example:

```powershell
$env:CICFLOWMETER_DOCKER_IMAGE = "<installed-image-name-or-id>"
python src/extraction/run_all_extractions.py
```

The scripts verify that the configured Docker image exists before starting CICFlowMeter.

## External Requirements

Re-running raw extraction requires the original raw captures and metadata plus the extraction tooling. In particular:

- Docker
- a compatible CICFlowMeter Docker image
- Wireshark command-line tools used by the pipeline, including `editcap` and `reordercap`
- Python dependencies used by the extraction scripts, including `pandas` and `dpkt`

The corresponding raw inputs are expected under the paths defined in `pipeline_config.py`, including:

```text
data/raw/dataset1/
data/raw/dataset2/
data/raw/dataset3/
data/metadata/
```

These raw inputs are not required to reproduce the final reported model results when the finalized `data/ingested/` files are provided.

## Setup Helper

To create the expected extraction-side directory structure:

```bash
python src/extraction/setup_project_folders.py
```

This creates the raw-input, metadata, extraction-output, and temporary work directories expected by the extraction code.

## Submission Scope

The final model-training and evaluation workflow starts from:

```text
data/ingested/
```

Therefore, this extraction directory is retained as provenance/source code rather than as a prerequisite that must be executed before the final models.
