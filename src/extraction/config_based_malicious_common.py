import csv
import json
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


def resolve_config_path(primary_path: Path) -> Path:
    if primary_path.exists():
        return primary_path

    raise FileNotFoundError(
        f"Could not find mta_config.json at:\n{primary_path}"
    )


def load_mta_config(
    config_path: Path,
    print_grouping_check: bool = False,
) -> dict[str, set[tuple[str, int]]]:
    print(f"[*] Loading MTA config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    config_by_file: dict[str, set[tuple[str, int]]] = {}

    for entry in entries:
        pcap_file = str(entry["pcap_file"]).strip()
        c2_ip = str(entry["c2_ip"]).strip()
        c2_port = int(entry["c2_port"])
        config_by_file.setdefault(pcap_file, set()).add((c2_ip, c2_port))

    total_endpoints = sum(len(v) for v in config_by_file.values())

    print(f"[*] Config PCAP files: {len(config_by_file):,}")
    print(f"[*] Config C2 endpoints: {total_endpoints:,}")

    if print_grouping_check:
        print("\n[*] Config grouping check:")
        for pcap_name, endpoints in sorted(config_by_file.items()):
            if len(endpoints) > 1:
                endpoint_str = ", ".join(
                    f"{ip}:{port}" for ip, port in sorted(endpoints)
                )
                print(f"    {pcap_name} -> {endpoint_str}")

    return config_by_file


def get_config_for_raw_file(
    raw_file: Path,
    config_by_file: dict[str, set[tuple[str, int]]],
    exact_match_short_circuit: bool,
) -> set[tuple[str, int]]:
    """
    Preserve the two original scripts' tiny matching difference:
      - Dataset 2: exact filename match returns immediately.
      - Dataset 3: exact filename and all matching stems are unioned.
    """
    if exact_match_short_circuit and raw_file.name in config_by_file:
        return set(config_by_file[raw_file.name])

    endpoints: set[tuple[str, int]] = set()

    if raw_file.name in config_by_file:
        endpoints.update(config_by_file[raw_file.name])

    for config_name, config_endpoints in config_by_file.items():
        if Path(config_name).stem == raw_file.stem:
            endpoints.update(config_endpoints)

    return endpoints


def is_configured_c2_packet(
    buf: bytes,
    datalink: int,
    c2_endpoints: set[tuple[str, int]],
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

        src_endpoint = (src_ip, int(tcp.sport))
        dst_endpoint = (dst_ip, int(tcp.dport))

        return src_endpoint in c2_endpoints or dst_endpoint in c2_endpoints

    except Exception:
        return False


def filter_raw_pcap_to_configured_c2(
    raw_pcap: Path,
    filtered_pcap: Path,
    c2_endpoints: set[tuple[str, int]],
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

                if is_configured_c2_packet(buf, datalink, c2_endpoints):
                    writer.writepkt(buf, ts)
                    kept_packets += 1

    return total_packets, kept_packets


def filter_and_reorder_one_file(
    raw_file_str: str,
    output_dir_str: str,
    c2_endpoints: set[tuple[str, int]],
) -> tuple[str, int, int, bool, str]:
    raw_file = Path(raw_file_str)
    output_dir = Path(output_dir_str)

    filtered_unordered = output_dir / f"{raw_file.stem}_configured_c2_unordered.pcap"
    final_ordered = output_dir / f"{raw_file.stem}_configured_c2.pcap"

    try:
        total_packets, kept_packets = filter_raw_pcap_to_configured_c2(
            raw_file,
            filtered_unordered,
            c2_endpoints,
        )

        if kept_packets == 0:
            if filtered_unordered.exists():
                filtered_unordered.unlink()
            return raw_file.name, total_packets, kept_packets, False, "no configured C2 packets kept"

        reorder_message = reorder_pcap(filtered_unordered, final_ordered)

        if filtered_unordered.exists():
            filtered_unordered.unlink()

        endpoint_str = ", ".join(f"{ip}:{port}" for ip, port in sorted(c2_endpoints))

        return (
            raw_file.name,
            total_packets,
            kept_packets,
            True,
            f"{reorder_message} | endpoints={endpoint_str}",
        )

    except Exception as exc:
        return raw_file.name, 0, 0, False, str(exc)


def filter_rows_by_config(
    chunk: pd.DataFrame,
    c2_endpoints: set[tuple[str, int]],
) -> pd.DataFrame:
    src_ip = chunk["Src IP"].astype(str)
    dst_ip = chunk["Dst IP"].astype(str)

    src_port = pd.to_numeric(chunk["Src Port"], errors="coerce")
    dst_port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    protocol = pd.to_numeric(chunk["Protocol"], errors="coerce")

    endpoint_mask = pd.Series(False, index=chunk.index)

    for c2_ip, c2_port in c2_endpoints:
        c2_port = int(c2_port)
        endpoint_mask |= (
            ((src_ip == c2_ip) & (src_port == c2_port))
            | ((dst_ip == c2_ip) & (dst_port == c2_port))
        )

    return chunk[(protocol == 6) & endpoint_mask].copy()


def get_job_base_name_from_csv(csv_file: Path) -> str:
    job_name = csv_file.parent.parent.name
    suffix = "_configured_c2"
    if job_name.endswith(suffix):
        return job_name[:-len(suffix)]
    return job_name


def find_config_endpoints_by_base_name(
    base_name: str,
    config_by_file: dict[str, set[tuple[str, int]]],
) -> set[tuple[str, int]]:
    endpoints: set[tuple[str, int]] = set()

    for config_name, config_endpoints in config_by_file.items():
        if Path(config_name).stem == base_name:
            endpoints.update(config_endpoints)

    return endpoints


def align_filter_and_label_csv_append(
    input_csv: Path,
    output_csv: Path,
    reference_cols: list[str],
    c2_endpoints: set[tuple[str, int]],
    write_header: bool,
    label_value: str,
    chunksize: int,
) -> tuple[int, int, bool]:
    total_input_rows = 0
    total_kept_rows = 0
    wrote_any_rows = False

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        chunk = normalize_columns(chunk)
        total_input_rows += len(chunk)

        required_cols = {"Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"}
        missing_filter = required_cols - set(chunk.columns)

        if missing_filter:
            raise RuntimeError(
                f"{input_csv.name} missing filter columns: {missing_filter}"
            )

        chunk = filter_rows_by_config(chunk, c2_endpoints)

        if chunk.empty:
            continue

        chunk["Label"] = label_value

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
    config_by_file: dict[str, set[tuple[str, int]]],
    final_output_dir: Path,
    label_value: str,
    chunksize: int,
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
        c2_endpoints = find_config_endpoints_by_base_name(base_name, config_by_file)

        if not c2_endpoints:
            print(f"[WARNING] No config found for: {base_name}")
            continue

        output_csv = final_output_dir / f"{base_name}.csv"

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
                c2_endpoints=c2_endpoints,
                write_header=write_header,
                label_value=label_value,
                chunksize=chunksize,
            )

            group_input_rows += input_rows
            group_kept_rows += kept_rows

            if wrote_rows:
                write_header = False

        if group_kept_rows == 0:
            if output_csv.exists():
                output_csv.unlink()
            print(f"[WARNING] {base_name}: no configured C2 rows kept")
        else:
            endpoint_str = ", ".join(
                f"{ip}:{port}" for ip, port in sorted(c2_endpoints)
            )
            print(
                f"[OK] {base_name}: "
                f"CICFlowMeter rows={group_input_rows:,}, "
                f"configured C2 rows kept={group_kept_rows:,}, "
                f"endpoints={endpoint_str}"
            )

        total_kept += group_kept_rows

    return total_kept


def run_config_based_malicious_extraction(
    *,
    dataset_number: int,
    input_dir: Path,
    mta_config_json: Path,
    run_root: Path,
    cic_input_dir: Path,
    cic_output_dir: Path,
    parallel_cic_root: Path,
    final_output_dir: Path,
    reference_schema_dir: Path,
    docker_image: str,
    label_value: str,
    chunksize: int,
    max_filter_workers: int,
    max_cicflowmeter_workers: int,
    clean_outputs: bool,
    exact_match_short_circuit: bool,
    print_config_grouping_check: bool,
) -> None:
    verify_docker_image(docker_image)
    verify_tool("reordercap")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    config_path = resolve_config_path(mta_config_json)
    config_by_file = load_mta_config(
        config_path,
        print_grouping_check=print_config_grouping_check,
    )
    reference_cols = load_reference_schema(reference_schema_dir)

    run_root.mkdir(parents=True, exist_ok=True)

    if clean_outputs:
        clean_dir(cic_input_dir)
        clean_dir(cic_output_dir)
        clean_dir(parallel_cic_root)
        clean_dir(final_output_dir)
    else:
        cic_input_dir.mkdir(parents=True, exist_ok=True)
        cic_output_dir.mkdir(parents=True, exist_ok=True)
        parallel_cic_root.mkdir(parents=True, exist_ok=True)
        final_output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([
        file for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in {".pcap", ".pcapng"}
    ])

    print("\n" + "=" * 80)
    print(f"PARALLEL DATASET {dataset_number} MALICIOUS CONFIG-BASED EXTRACTION")
    print("=" * 80)
    print(f"Input PCAP dir:       {input_dir}")
    print(f"MTA config:           {config_path}")
    print(f"Output dir:           {final_output_dir}")
    print(f"Docker image:         {docker_image}")
    print(f"Raw PCAP files:       {len(raw_files)}")
    print(f"Filter workers:       {max_filter_workers}")
    print(f"CICFlowMeter workers: {max_cicflowmeter_workers}")
    print(f"Shared columns:       {len(reference_cols)}")

    if not raw_files:
        print("No PCAP/PCAPNG files found.")
        return

    print("\n[Step 1] Filtering and reordering malicious PCAPs by config...")

    kept_pcap_count = 0
    skipped_no_config = 0

    with ProcessPoolExecutor(max_workers=max_filter_workers) as executor:
        futures = []

        for raw_file in raw_files:
            c2_endpoints = get_config_for_raw_file(
                raw_file,
                config_by_file,
                exact_match_short_circuit=exact_match_short_circuit,
            )

            if not c2_endpoints:
                skipped_no_config += 1
                print(f"[SKIP] No config entry for: {raw_file.name}")
                continue

            futures.append(
                executor.submit(
                    filter_and_reorder_one_file,
                    str(raw_file),
                    str(cic_input_dir),
                    c2_endpoints,
                )
            )

        for future in as_completed(futures):
            name, total_packets, kept_packets, success, message = future.result()

            if success:
                kept_pcap_count += 1
                print(
                    f"[OK] {name}: packets={total_packets:,}, "
                    f"configured C2 packets kept={kept_packets:,}"
                )
                if message:
                    print(f"     {message}")
            else:
                print(
                    f"[SKIP/ERROR] {name}: packets={total_packets:,}, "
                    f"kept={kept_packets:,}, reason={message}"
                )

    print(f"[*] PCAPs skipped because no config entry: {skipped_no_config}")

    if kept_pcap_count == 0:
        print("No filtered malicious PCAPs were created. Stopping.")
        return

    print(f"[*] Filtered ordered malicious PCAPs ready: {kept_pcap_count}")

    output_dirs = run_cicflowmeter_parallel(
        cic_input_dir,
        parallel_cic_root,
        docker_image,
        max_cicflowmeter_workers,
    )

    total_rows = align_all_parallel_outputs(
        output_dirs,
        reference_cols,
        config_by_file,
        final_output_dir,
        label_value,
        chunksize,
    )

    print("\n" + "=" * 80)
    print(f"PARALLEL DATASET {dataset_number} MALICIOUS CONFIG-BASED EXTRACTION DONE")
    print("=" * 80)
    print("Aligned malicious C2 CSVs saved to:")
    print(final_output_dir)
    print(f"Total malicious configured C2 rows kept: {total_rows:,}")
