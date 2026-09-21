import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dpkt
import pandas as pd

from cicflowmeter_common import (
    clean_dir,
    extract_ip_layer,
    inet_to_str,
    load_reference_schema,
    normalize_columns,
    open_pcap_reader,
    reorder_pcap,
    run_cicflowmeter_parallel,
    verify_docker_image,
    verify_tool,
)


from pipeline_config import (
    CICFLOWMETER_DOCKER_IMAGE,
    CIC_INPUT_DIR,
    CIC_OUTPUT_DIR,
    DATASET3_BENIGN_INPUT_DIR,
    DATASET3_BENIGN_OUTPUT_DIR,
    DATASET3_BENIGN_PARALLEL_CIC_ROOT,
    REFERENCE_SCHEMA_DIR,
    RUN_ROOT,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = DATASET3_BENIGN_INPUT_DIR
PARALLEL_CIC_ROOT = DATASET3_BENIGN_PARALLEL_CIC_ROOT
FINAL_OUTPUT_DIR = DATASET3_BENIGN_OUTPUT_DIR


DOCKER_IMAGE = CICFLOWMETER_DOCKER_IMAGE
LABEL_VALUE = "Benign"
CHUNKSIZE = 300_000
MAX_FILTER_WORKERS = 3
MAX_CICFLOWMETER_WORKERS = 2
CLEAN_OUTPUTS = True


def get_target_ip_for_file(raw_file: Path) -> str:
    """Kali files use 192.168.1.191; all other files use 10.0.2.15."""
    if "kali" in raw_file.name.lower():
        return "192.168.1.191"
    return "10.0.2.15"


def is_target_https_packet(
    buf: bytes,
    datalink: int,
    target_ip: str,
) -> bool:
    try:
        ip_layer = extract_ip_layer(buf, datalink, allow_linux_sll=True)

        if not isinstance(ip_layer, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return False

        if not isinstance(ip_layer.data, dpkt.tcp.TCP):
            return False

        tcp = ip_layer.data
        src_ip = inet_to_str(ip_layer.src)
        dst_ip = inet_to_str(ip_layer.dst)

        target_involved = src_ip == target_ip or dst_ip == target_ip
        https_port = int(tcp.sport) == 443 or int(tcp.dport) == 443

        return target_involved and https_port

    except Exception:
        return False


def filter_raw_pcap_to_target_https(
    raw_pcap: Path,
    filtered_pcap: Path,
    target_ip: str,
) -> tuple[int, int]:
    total_packets = 0
    kept_packets = 0

    with raw_pcap.open("rb") as f_in:
        reader = open_pcap_reader(f_in)
        datalink = reader.datalink()

        with filtered_pcap.open("wb") as f_out:
            writer = dpkt.pcap.Writer(f_out, linktype=datalink)

            for ts, buf in reader:
                total_packets += 1

                if is_target_https_packet(buf, datalink, target_ip):
                    writer.writepkt(buf, ts)
                    kept_packets += 1

    return total_packets, kept_packets


def filter_and_reorder_one_file(
    raw_file_str: str,
    output_dir_str: str,
) -> tuple[str, str, int, int, bool, str]:
    raw_file = Path(raw_file_str)
    output_dir = Path(output_dir_str)
    target_ip = get_target_ip_for_file(raw_file)

    filtered_unordered = output_dir / f"{raw_file.stem}_benign_https_unordered.pcap"
    final_ordered = output_dir / f"{raw_file.stem}_benign_https.pcap"

    try:
        total_packets, kept_packets = filter_raw_pcap_to_target_https(
            raw_file,
            filtered_unordered,
            target_ip,
        )

        if kept_packets == 0:
            if filtered_unordered.exists():
                filtered_unordered.unlink()
            return (
                raw_file.name,
                target_ip,
                total_packets,
                kept_packets,
                False,
                "no target HTTPS packets kept",
            )

        reorder_message = reorder_pcap(filtered_unordered, final_ordered)

        if filtered_unordered.exists():
            filtered_unordered.unlink()

        return (
            raw_file.name,
            target_ip,
            total_packets,
            kept_packets,
            True,
            reorder_message,
        )

    except Exception as exc:
        return raw_file.name, target_ip, 0, 0, False, str(exc)


def get_job_base_name_from_csv(csv_file: Path) -> str:
    job_name = csv_file.parent.parent.name
    suffix = "_benign_https"
    if job_name.endswith(suffix):
        return job_name[:-len(suffix)]
    return job_name


def filter_rows_by_target_https(
    chunk: pd.DataFrame,
    target_ip: str,
) -> pd.DataFrame:
    src_ip = chunk["Src IP"].astype(str)
    dst_ip = chunk["Dst IP"].astype(str)

    src_port = pd.to_numeric(chunk["Src Port"], errors="coerce")
    dst_port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    protocol = pd.to_numeric(chunk["Protocol"], errors="coerce")

    target_involved = (src_ip == target_ip) | (dst_ip == target_ip)
    https_port = (src_port == 443) | (dst_port == 443)

    return chunk[(protocol == 6) & target_involved & https_port].copy()


def align_filter_and_label_csv_append(
    input_csv: Path,
    output_csv: Path,
    reference_cols: list[str],
    target_ip: str,
    write_header: bool,
) -> tuple[int, int, bool]:
    total_input_rows = 0
    total_kept_rows = 0
    wrote_any_rows = False

    for chunk in pd.read_csv(input_csv, chunksize=CHUNKSIZE, low_memory=False):
        chunk = normalize_columns(chunk)
        total_input_rows += len(chunk)

        required_cols = {"Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"}
        missing = required_cols - set(chunk.columns)

        if missing:
            raise RuntimeError(
                f"{input_csv.name} missing filter columns: {missing}"
            )

        chunk = filter_rows_by_target_https(chunk, target_ip)

        if chunk.empty:
            continue

        chunk["Label"] = LABEL_VALUE

        missing_schema = [col for col in reference_cols if col not in chunk.columns]
        if missing_schema:
            raise RuntimeError(
                f"{input_csv.name} missing shared schema columns:\n"
                + "\n".join(missing_schema)
            )

        chunk = chunk[reference_cols]

        chunk.to_csv(
            output_csv,
            mode="a",
            index=False,
            header=write_header and not wrote_any_rows,
            quoting=csv.QUOTE_MINIMAL,
        )

        wrote_any_rows = True
        total_kept_rows += len(chunk)

    return total_input_rows, total_kept_rows, wrote_any_rows


def align_all_parallel_outputs(
    output_dirs: list[Path],
    reference_cols: list[str],
) -> int:
    csvs_by_base_name: dict[str, list[Path]] = {}

    for output_dir in output_dirs:
        for csv_file in sorted(output_dir.rglob("*.csv")):
            base_name = get_job_base_name_from_csv(csv_file)
            csvs_by_base_name.setdefault(base_name, []).append(csv_file)

    if not csvs_by_base_name:
        raise FileNotFoundError("No CICFlowMeter CSV files found in parallel output folders.")

    print("\n[Step 3] Aligning and labeling CICFlowMeter CSVs...")
    print(f"[*] Original PCAP groups found: {len(csvs_by_base_name)}")

    total_kept = 0

    for base_name, csv_files in sorted(csvs_by_base_name.items()):
        fake_raw_path = Path(base_name)
        target_ip = get_target_ip_for_file(fake_raw_path)
        output_csv = FINAL_OUTPUT_DIR / f"{base_name}.csv"

        if output_csv.exists():
            output_csv.unlink()

        group_input_rows = 0
        group_kept_rows = 0
        write_header = True

        for csv_file in csv_files:
            input_rows, kept_rows, wrote_rows = align_filter_and_label_csv_append(
                input_csv=csv_file,
                output_csv=output_csv,
                reference_cols=reference_cols,
                target_ip=target_ip,
                write_header=write_header,
            )

            group_input_rows += input_rows
            group_kept_rows += kept_rows

            if wrote_rows:
                write_header = False

        if group_kept_rows == 0:
            if output_csv.exists():
                output_csv.unlink()
            print(f"[WARNING] {base_name}: no benign HTTPS rows kept")
        else:
            print(
                f"[OK] {base_name}: "
                f"target_ip={target_ip}, "
                f"CICFlowMeter rows={group_input_rows:,}, "
                f"benign HTTPS rows kept={group_kept_rows:,}"
            )

        total_kept += group_kept_rows

    return total_kept


def main() -> None:
    verify_docker_image(DOCKER_IMAGE)
    verify_tool("reordercap")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    reference_cols = load_reference_schema(REFERENCE_SCHEMA_DIR)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUTS:
        clean_dir(CIC_INPUT_DIR)
        clean_dir(CIC_OUTPUT_DIR)
        clean_dir(PARALLEL_CIC_ROOT)
        clean_dir(FINAL_OUTPUT_DIR)
    else:
        CIC_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        CIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PARALLEL_CIC_ROOT.mkdir(parents=True, exist_ok=True)
        FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([
        file for file in INPUT_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in {".pcap", ".pcapng"}
    ])

    print("\n" + "=" * 80)
    print("PARALLEL DATASET 3 BENIGN CTU HTTPS EXTRACTION")
    print("=" * 80)
    print(f"Input PCAP dir:       {INPUT_DIR}")
    print(f"Output dir:           {FINAL_OUTPUT_DIR}")
    print(f"Docker image:         {DOCKER_IMAGE}")
    print(f"Raw PCAP files:       {len(raw_files)}")
    print(f"Filter workers:       {MAX_FILTER_WORKERS}")
    print(f"CICFlowMeter workers: {MAX_CICFLOWMETER_WORKERS}")
    print(f"Shared columns:       {len(reference_cols)}")
    print("Target rule:")
    print("  kali files       -> 192.168.1.191")
    print("  win/normal files -> 10.0.2.15")

    if not raw_files:
        print("No PCAP/PCAPNG files found.")
        return

    print("\n[Step 1] Filtering and reordering benign PCAPs...")

    kept_pcap_count = 0

    with ProcessPoolExecutor(max_workers=MAX_FILTER_WORKERS) as executor:
        futures = [
            executor.submit(
                filter_and_reorder_one_file,
                str(raw_file),
                str(CIC_INPUT_DIR),
            )
            for raw_file in raw_files
        ]

        for future in as_completed(futures):
            name, target_ip, total_packets, kept_packets, success, message = future.result()

            if success:
                kept_pcap_count += 1
                print(
                    f"[OK] {name}: target_ip={target_ip}, "
                    f"packets={total_packets:,}, "
                    f"HTTPS packets kept={kept_packets:,}"
                )
                if message:
                    print(f"     reordercap: {message}")
            else:
                print(
                    f"[SKIP/ERROR] {name}: target_ip={target_ip}, "
                    f"packets={total_packets:,}, "
                    f"kept={kept_packets:,}, reason={message}"
                )

    if kept_pcap_count == 0:
        print("No filtered benign PCAPs were created. Stopping.")
        return

    print(f"[*] Filtered ordered benign PCAPs ready: {kept_pcap_count}")

    output_dirs = run_cicflowmeter_parallel(
        CIC_INPUT_DIR,
        PARALLEL_CIC_ROOT,
        DOCKER_IMAGE,
        MAX_CICFLOWMETER_WORKERS,
    )

    total_rows = align_all_parallel_outputs(output_dirs, reference_cols)

    print("\n" + "=" * 80)
    print("PARALLEL DATASET 3 BENIGN CTU HTTPS EXTRACTION DONE")
    print("=" * 80)
    print("Aligned benign HTTPS CSVs saved to:")
    print(FINAL_OUTPUT_DIR)
    print(f"Total benign HTTPS rows kept: {total_rows:,}")


if __name__ == "__main__":
    main()
