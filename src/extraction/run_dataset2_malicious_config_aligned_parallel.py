from pathlib import Path

from config_based_malicious_common import run_config_based_malicious_extraction


from pipeline_config import (
    CICFLOWMETER_DOCKER_IMAGE,
    CIC_INPUT_DIR,
    CIC_OUTPUT_DIR,
    DATASET2_MALICIOUS_INPUT_DIR,
    DATASET2_MALICIOUS_OUTPUT_DIR,
    DATASET2_MALICIOUS_PARALLEL_CIC_ROOT,
    MTA_CONFIG_JSON,
    REFERENCE_SCHEMA_DIR,
    RUN_ROOT,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = DATASET2_MALICIOUS_INPUT_DIR
PARALLEL_CIC_ROOT = DATASET2_MALICIOUS_PARALLEL_CIC_ROOT
FINAL_OUTPUT_DIR = DATASET2_MALICIOUS_OUTPUT_DIR


DOCKER_IMAGE = CICFLOWMETER_DOCKER_IMAGE
LABEL_VALUE = "Malicious"
CHUNKSIZE = 300_000
MAX_FILTER_WORKERS = 3
MAX_CICFLOWMETER_WORKERS = 2
CLEAN_OUTPUTS = True


def main() -> None:
    run_config_based_malicious_extraction(
        dataset_number=2,
        input_dir=INPUT_DIR,
        mta_config_json=MTA_CONFIG_JSON,
        run_root=RUN_ROOT,
        cic_input_dir=CIC_INPUT_DIR,
        cic_output_dir=CIC_OUTPUT_DIR,
        parallel_cic_root=PARALLEL_CIC_ROOT,
        final_output_dir=FINAL_OUTPUT_DIR,
        reference_schema_dir=REFERENCE_SCHEMA_DIR,
        docker_image=DOCKER_IMAGE,
        label_value=LABEL_VALUE,
        chunksize=CHUNKSIZE,
        max_filter_workers=MAX_FILTER_WORKERS,
        max_cicflowmeter_workers=MAX_CICFLOWMETER_WORKERS,
        clean_outputs=CLEAN_OUTPUTS,
        # Preserve the original Dataset 2 exact-filename short-circuit.
        exact_match_short_circuit=True,
        print_config_grouping_check=False,
    )


if __name__ == "__main__":
    main()
