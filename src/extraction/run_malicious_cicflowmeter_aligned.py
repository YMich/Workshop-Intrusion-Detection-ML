import csv
from pathlib import Path

import pandas as pd

from cicflowmeter_common import (
    clean_dir,
    load_reference_schema,
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
    DATASET1_MALICIOUS_INPUT_DIR,
    DATASET1_MALICIOUS_OUTPUT_DIR,
    REFERENCE_SCHEMA_DIR,
    RUN_ROOT,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = DATASET1_MALICIOUS_INPUT_DIR
FINAL_OUTPUT_DIR = DATASET1_MALICIOUS_OUTPUT_DIR


DOCKER_IMAGE = CICFLOWMETER_DOCKER_IMAGE
LABEL_VALUE = "Malicious"
CHUNKSIZE = 300_000
CLEAN_FINAL_OUTPUT_DIR = True


def filter_https_tcp(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep only Protocol=TCP and flows using source or destination port 443."""
    src_port = pd.to_numeric(chunk["Src Port"], errors="coerce")
    dst_port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    protocol = pd.to_numeric(chunk["Protocol"], errors="coerce")

    is_tcp = protocol == 6
    is_https = (src_port == 443) | (dst_port == 443)

    return chunk[is_tcp & is_https].copy()


def align_filter_and_label_csv(
    input_csv: Path,
    output_csv: Path,
    reference_cols: list[str],
) -> tuple[int, int]:
    if output_csv.exists():
        output_csv.unlink()

    total_input_rows = 0
    total_kept_rows = 0
    wrote_header = False

    for chunk in pd.read_csv(input_csv, chunksize=CHUNKSIZE, low_memory=False):
        chunk = normalize_columns(chunk)
        total_input_rows += len(chunk)

        missing_before_filter = [
            col for col in ["Src Port", "Dst Port", "Protocol"]
            if col not in chunk.columns
        ]

        if missing_before_filter:
            raise RuntimeError(
                f"{input_csv.name} is missing required filter columns:\n"
                + "\n".join(missing_before_filter)
            )

        chunk = filter_https_tcp(chunk)

        if chunk.empty:
            continue

        chunk["Label"] = LABEL_VALUE

        missing_schema_cols = [col for col in reference_cols if col not in chunk.columns]
        if missing_schema_cols:
            raise RuntimeError(
                f"{input_csv.name} is missing required shared schema columns:\n"
                + "\n".join(missing_schema_cols)
            )

        chunk = chunk[reference_cols]

        chunk.to_csv(
            output_csv,
            mode="a",
            index=False,
            header=not wrote_header,
            quoting=csv.QUOTE_MINIMAL,
        )

        wrote_header = True
        total_kept_rows += len(chunk)

    return total_input_rows, total_kept_rows


def process_single_pcap(raw_file: Path, reference_cols: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"[+] Processing malicious file: {raw_file.name}")
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

    file_input_rows = 0
    file_kept_rows = 0

    for i, csv_file in enumerate(csv_files, start=1):
        if len(csv_files) == 1:
            output_csv = FINAL_OUTPUT_DIR / f"{raw_file.stem}.csv"
        else:
            output_csv = FINAL_OUTPUT_DIR / f"{raw_file.stem}_{i}.csv"

        input_rows, kept_rows = align_filter_and_label_csv(
            csv_file,
            output_csv,
            reference_cols,
        )

        file_input_rows += input_rows
        file_kept_rows += kept_rows

        if kept_rows == 0:
            if output_csv.exists():
                output_csv.unlink()
            print(f"[WARNING] No TCP/443 rows kept for: {raw_file.name}")
        else:
            print(f"[OK] Saved aligned HTTPS malicious CSV: {output_csv}")
            print(f"     Input rows: {input_rows:,}")
            print(f"     TCP/443 rows kept: {kept_rows:,}")

    print(f"[DONE] {raw_file.name}")
    print(f"       Total CICFlowMeter rows: {file_input_rows:,}")
    print(f"       Total TCP/443 malicious rows kept: {file_kept_rows:,}")


def main() -> None:
    verify_docker_image(DOCKER_IMAGE)
    verify_tool("editcap")
    verify_tool("reordercap")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CIC_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_FINAL_OUTPUT_DIR:
        clean_dir(FINAL_OUTPUT_DIR)
    else:
        FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_cols = load_reference_schema(REFERENCE_SCHEMA_DIR)

    raw_files = sorted([
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".pcap", ".pcapng"}
    ])

    print(f"Found {len(raw_files)} malicious PCAP/PCAPNG files.")
    print(f"Input:        {INPUT_DIR}")
    print(f"Output:       {FINAL_OUTPUT_DIR}")
    print(f"Docker image: {DOCKER_IMAGE}")
    print(f"Shared cols:  {len(reference_cols)}")
    print("Filter:       Protocol == 6 AND (Src Port == 443 OR Dst Port == 443)")

    if not raw_files:
        print("No PCAP/PCAPNG files found.")
        return

    total_kept = 0

    for raw_file in raw_files:
        process_single_pcap(raw_file, reference_cols)

    for csv_file in FINAL_OUTPUT_DIR.glob("*.csv"):
        try:
            total_kept += sum(
                1 for _ in open(csv_file, "r", encoding="utf-8", errors="ignore")
            ) - 1
        except OSError:
            pass

    print("\n" + "=" * 80)
    print("MALICIOUS HTTPS EXTRACTION DONE")
    print("=" * 80)
    print("Aligned malicious HTTPS CSVs saved to:")
    print(FINAL_OUTPUT_DIR)
    print(f"Total malicious TCP/443 rows kept: {total_kept:,}")


if __name__ == "__main__":
    main()
