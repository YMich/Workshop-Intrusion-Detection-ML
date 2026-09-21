import csv
from pathlib import Path

import pandas as pd

from cicflowmeter_common import (
    SHARED_COLUMNS,
    clean_dir,
    normalize_columns,
    prepare_cic_input_file,
    run_cicflowmeter_serial,
    verify_docker_image,
    verify_tool,
)


from pipeline_config import (
    CICFLOWMETER_DOCKER_IMAGE,
    CIC_INPUT_DIR,
    CIC_OUTPUT_DIR,
    DATASET1_BENIGN_INPUT_DIR,
    DATASET1_BENIGN_OUTPUT_DIR,
    RUN_ROOT,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = DATASET1_BENIGN_INPUT_DIR
FINAL_OUTPUT_DIR = DATASET1_BENIGN_OUTPUT_DIR

DOCKER_IMAGE = CICFLOWMETER_DOCKER_IMAGE
LABEL_VALUE = "Benign"
CHUNKSIZE = 300_000


def align_and_label_csv(input_csv: Path, output_csv: Path) -> int:
    """Add Label=Benign and keep exactly the canonical shared schema."""
    if output_csv.exists():
        output_csv.unlink()

    total_rows = 0
    wrote_header = False

    for chunk in pd.read_csv(input_csv, chunksize=CHUNKSIZE, low_memory=False):
        chunk = normalize_columns(chunk)
        chunk["Label"] = LABEL_VALUE

        missing = [col for col in SHARED_COLUMNS if col not in chunk.columns]
        if missing:
            raise RuntimeError(
                f"{input_csv.name} is missing required shared columns:\n"
                + "\n".join(missing)
            )

        chunk = chunk[SHARED_COLUMNS]

        chunk.to_csv(
            output_csv,
            mode="a",
            index=False,
            header=not wrote_header,
            quoting=csv.QUOTE_MINIMAL,
        )

        wrote_header = True
        total_rows += len(chunk)

    return total_rows


def process_single_pcap(raw_file: Path) -> None:
    print("\n" + "=" * 80)
    print(f"[+] Processing benign file: {raw_file.name}")
    print("=" * 80)

    clean_dir(CIC_INPUT_DIR)
    clean_dir(CIC_OUTPUT_DIR)

    prepared_file = prepare_cic_input_file(raw_file, CIC_INPUT_DIR)
    print(f"[*] CICFlowMeter input file: {prepared_file.name}")

    run_cicflowmeter_serial(RUN_ROOT, DOCKER_IMAGE)

    csv_files = sorted(CIC_OUTPUT_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"[WARNING] No CSV produced for: {raw_file.name}")
        return

    total_rows = 0

    for i, csv_file in enumerate(csv_files, start=1):
        if len(csv_files) == 1:
            output_csv = FINAL_OUTPUT_DIR / f"{raw_file.stem}.csv"
        else:
            output_csv = FINAL_OUTPUT_DIR / f"{raw_file.stem}_{i}.csv"

        rows = align_and_label_csv(csv_file, output_csv)
        total_rows += rows

        print(f"[OK] Saved aligned CSV: {output_csv}")
        print(f"     Rows labeled Benign: {rows:,}")

    print(f"[DONE] {raw_file.name} | Total rows: {total_rows:,}")


def main() -> None:
    verify_docker_image(DOCKER_IMAGE)
    verify_tool("editcap")
    verify_tool("reordercap")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CIC_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".pcap", ".pcapng"}
    ])

    print(f"Found {len(raw_files)} benign PCAP/PCAPNG files.")
    print(f"Input:        {INPUT_DIR}")
    print(f"Output:       {FINAL_OUTPUT_DIR}")
    print(f"Docker image: {DOCKER_IMAGE}")
    print(f"Shared cols:  {len(SHARED_COLUMNS)}")

    if not raw_files:
        print("No PCAP/PCAPNG files found.")
        return

    for raw_file in raw_files:
        process_single_pcap(raw_file)

    print("\n" + "=" * 80)
    print("BENIGN EXTRACTION DONE")
    print("=" * 80)
    print("Aligned benign CSVs saved to:")
    print(FINAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
