import csv
import ipaddress
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
    DATASET2_BENIGN_INPUT_DIR,
    DATASET2_BENIGN_OUTPUT_DIR,
    DATASET2_BENIGN_PARALLEL_CIC_ROOT,
    HIKARI_ALLFLOWMETER_CSV,
    METADATA_ROOT,
    REFERENCE_SCHEMA_DIR,
    RUN_ROOT,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = DATASET2_BENIGN_INPUT_DIR
PARALLEL_CIC_ROOT = DATASET2_BENIGN_PARALLEL_CIC_ROOT
FINAL_OUTPUT_DIR = DATASET2_BENIGN_OUTPUT_DIR


DOCKER_IMAGE = CICFLOWMETER_DOCKER_IMAGE
LABEL_VALUE = "Benign"
CHUNKSIZE = 300_000
MAX_FILTER_WORKERS = 10
MAX_CICFLOWMETER_WORKERS = 10
CLEAN_OUTPUTS = True


def resolve_hikari_csv() -> Path:
    candidates = [
        HIKARI_ALLFLOWMETER_CSV,
        Path(str(HIKARI_ALLFLOWMETER_CSV) + ".csv"),
        METADATA_ROOT / "HIKARI2021" / "ALLFLOWMETER_HIKARI2021.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find ALLFLOWMETER_HIKARI2021 CSV. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def build_hikari_benign_public_ip_index(csv_path: Path) -> set[str]:
    print(f"[*] Loading HIKARI CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = {"traffic_category", "Label", "responh"}
    missing = required_cols - set(df.columns)

    if missing:
        raise RuntimeError(f"HIKARI CSV missing required columns: {missing}")

    is_benign_string = df["traffic_category"].astype(str).str.strip() == "Benign"
    is_benign_int = pd.to_numeric(df["Label"], errors="coerce") == 0

    benign_df = df[is_benign_string & is_benign_int]
    raw_ips = set(benign_df["responh"].dropna().astype(str).unique())

    public_ips: set[str] = set()

    for ip in raw_ips:
        try:
            ip_obj = ipaddress.ip_address(ip)
            if not ip_obj.is_private and not ip_obj.is_loopback:
                public_ips.add(ip)
        except ValueError:
            continue

    print(f"[*] Raw benign responder IPs: {len(raw_ips):,}")
    print(f"[*] Public benign responder IPs kept: {len(public_ips):,}")

    if not public_ips:
        raise RuntimeError("No public benign responder IPs found.")

    return public_ips


def is_benign_https_packet(
    buf: bytes,
    datalink: int,
    benign_public_ips: set[str],
) -> bool:
    try:
        # Preserve original Dataset 2 behavior: Ethernet + RAW only, no SLL.
        ip_layer = extract_ip_layer(buf, datalink, allow_linux_sll=False)

        if not isinstance(ip_layer, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return False

        if not isinstance(ip_layer.data, dpkt.tcp.TCP):
            return False

        tcp = ip_layer.data
        src_ip = inet_to_str(ip_layer.src)
        dst_ip = inet_to_str(ip_layer.dst)

        to_public_https = dst_ip in benign_public_ips and tcp.dport == 443
        from_public_https = src_ip in benign_public_ips and tcp.sport == 443

        return to_public_https or from_public_https

    except Exception:
        return False


def filter_raw_pcap_to_benign_https(
    raw_pcap: Path,
    filtered_pcap: Path,
    benign_public_ips: set[str],
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

                if is_benign_https_packet(buf, datalink, benign_public_ips):
                    writer.writepkt(buf, ts)
                    kept_packets += 1

    return total_packets, kept_packets


def filter_and_reorder_one_file(
    raw_file_str: str,
    output_dir_str: str,
    benign_public_ips: set[str],
) -> tuple[str, int, int, bool, str]:
    raw_file = Path(raw_file_str)
    output_dir = Path(output_dir_str)

    filtered_unordered = output_dir / f"{raw_file.stem}_benign_https_unordered.pcap"
    final_ordered = output_dir / f"{raw_file.stem}_benign_https.pcap"

    try:
        total_packets, kept_packets = filter_raw_pcap_to_benign_https(
            raw_file,
            filtered_unordered,
            benign_public_ips,
        )

        if kept_packets == 0:
            if filtered_unordered.exists():
                filtered_unordered.unlink()
            return raw_file.name, total_packets, kept_packets, False, "no benign HTTPS packets kept"

        reorder_message = reorder_pcap(filtered_unordered, final_ordered)

        if filtered_unordered.exists():
            filtered_unordered.unlink()

        return raw_file.name, total_packets, kept_packets, True, reorder_message

    except Exception as exc:
        return raw_file.name, 0, 0, False, str(exc)


def filter_https_tcp(chunk: pd.DataFrame) -> pd.DataFrame:
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

        required_filter_cols = {"Src Port", "Dst Port", "Protocol"}
        missing_filter = required_filter_cols - set(chunk.columns)

        if missing_filter:
            raise RuntimeError(
                f"{input_csv.name} missing filter columns: {missing_filter}"
            )

        chunk = filter_https_tcp(chunk)

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
            header=not wrote_header,
            quoting=csv.QUOTE_MINIMAL,
        )

        wrote_header = True
        total_kept_rows += len(chunk)

    return total_input_rows, total_kept_rows


def align_all_parallel_outputs(
    output_dirs: list[Path],
    reference_cols: list[str],
) -> int:
    all_csv_files: list[Path] = []

    for output_dir in output_dirs:
        all_csv_files.extend(sorted(output_dir.rglob("*.csv")))

    if not all_csv_files:
        raise FileNotFoundError("No CICFlowMeter CSV files found in parallel output folders.")

    print("\n[Step 3] Aligning and labeling CICFlowMeter CSVs...")
    print(f"[*] CICFlowMeter CSV files found: {len(all_csv_files)}")

    total_kept = 0

    for csv_file in all_csv_files:
        # Preserve original Dataset 2 output naming (job folder name, including suffix).
        job_name = csv_file.parent.parent.name
        output_csv = FINAL_OUTPUT_DIR / f"{job_name}.csv"

        input_rows, kept_rows = align_filter_and_label_csv(
            csv_file,
            output_csv,
            reference_cols,
        )

        if kept_rows == 0:
            if output_csv.exists():
                output_csv.unlink()
            print(f"[WARNING] {job_name}: no TCP/443 rows kept")
        else:
            print(
                f"[OK] {job_name}: "
                f"CICFlowMeter rows={input_rows:,}, kept={kept_rows:,}"
            )

        total_kept += kept_rows

    return total_kept


def main() -> None:
    verify_docker_image(DOCKER_IMAGE)
    verify_tool("reordercap")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    hikari_csv = resolve_hikari_csv()
    benign_public_ips = build_hikari_benign_public_ip_index(hikari_csv)
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
    print("PARALLEL DATASET 2 HIKARI BENIGN HTTPS EXTRACTION")
    print("=" * 80)
    print(f"Input PCAP dir:      {INPUT_DIR}")
    print(f"HIKARI CSV:          {hikari_csv}")
    print(f"Output dir:          {FINAL_OUTPUT_DIR}")
    print(f"Docker image:        {DOCKER_IMAGE}")
    print(f"Raw PCAP files:      {len(raw_files)}")
    print(f"Filter workers:      {MAX_FILTER_WORKERS}")
    print(f"CICFlowMeter workers:{MAX_CICFLOWMETER_WORKERS}")
    print(f"Shared columns:      {len(reference_cols)}")

    if not raw_files:
        print("No PCAP/PCAPNG files found.")
        return

    print("\n[Step 1] Filtering and reordering raw PCAPs in parallel...")

    kept_pcap_count = 0

    with ProcessPoolExecutor(max_workers=MAX_FILTER_WORKERS) as executor:
        futures = [
            executor.submit(
                filter_and_reorder_one_file,
                str(raw_file),
                str(CIC_INPUT_DIR),
                benign_public_ips,
            )
            for raw_file in raw_files
        ]

        for future in as_completed(futures):
            name, total_packets, kept_packets, success, message = future.result()

            if success:
                kept_pcap_count += 1
                print(
                    f"[OK] {name}: packets={total_packets:,}, "
                    f"kept={kept_packets:,}"
                )
                if message:
                    print(f"     reordercap: {message}")
            else:
                print(
                    f"[SKIP/ERROR] {name}: packets={total_packets:,}, "
                    f"kept={kept_packets:,}, reason={message}"
                )

    if kept_pcap_count == 0:
        print("No filtered PCAPs were created. Stopping.")
        return

    print(f"[*] Filtered ordered PCAPs ready: {kept_pcap_count}")

    output_dirs = run_cicflowmeter_parallel(
        CIC_INPUT_DIR,
        PARALLEL_CIC_ROOT,
        DOCKER_IMAGE,
        MAX_CICFLOWMETER_WORKERS,
    )

    total_rows = align_all_parallel_outputs(output_dirs, reference_cols)

    print("\n" + "=" * 80)
    print("PARALLEL DATASET 2 HIKARI BENIGN HTTPS EXTRACTION DONE")
    print("=" * 80)
    print("Aligned benign HTTPS CSVs saved to:")
    print(FINAL_OUTPUT_DIR)
    print(f"Total benign HTTPS rows kept: {total_rows:,}")


if __name__ == "__main__":
    main()
