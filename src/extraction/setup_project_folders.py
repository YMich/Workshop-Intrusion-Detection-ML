from pipeline_config import (
    DATASET1_BENIGN_INPUT_DIR,
    DATASET1_MALICIOUS_INPUT_DIR,
    DATASET2_BENIGN_INPUT_DIR,
    DATASET2_MALICIOUS_INPUT_DIR,
    DATASET3_BENIGN_INPUT_DIR,
    DATASET3_MALICIOUS_INPUT_DIR,
    EXTRACTED_ROOT,
    METADATA_ROOT,
    PARALLEL_ROOT,
    PROJECT_ROOT,
    RUN_ROOT,
)


def main() -> None:
    directories = [
        DATASET1_BENIGN_INPUT_DIR,
        DATASET1_MALICIOUS_INPUT_DIR,
        DATASET2_BENIGN_INPUT_DIR,
        DATASET2_MALICIOUS_INPUT_DIR,
        DATASET3_BENIGN_INPUT_DIR,
        DATASET3_MALICIOUS_INPUT_DIR,
        METADATA_ROOT,
        EXTRACTED_ROOT,
        RUN_ROOT,
        PARALLEL_ROOT,
        PROJECT_ROOT / "src" / "extraction",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "report",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {directory}")

    print("\nFolder structure created.")
    print(r"Now copy the raw PCAP/PCAPNG files and metadata into the matching data folders. Place friday.csv directly in data\raw\dataset1\benign.")


if __name__ == "__main__":
    main()
