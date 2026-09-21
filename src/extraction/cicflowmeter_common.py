import shutil
import socket
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dpkt
import pandas as pd


# ============================================================
# CANONICAL CICFLOWMETER SCHEMA
# ============================================================

# Same schema as CICIDS2017 after removing id/Attempted Category
# and upgraded-only columns missing from CICIDS2017.
SHARED_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd RST Flags",
    "Bwd RST Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "ICMP Code",
    "ICMP Type",
    "Total Connection Flow Time",
    "Label",
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

RENAME_MAP = {
    "Source IP": "Src IP",
    "Destination IP": "Dst IP",
    "Source Port": "Src Port",
    "Destination Port": "Dst Port",

    "Total Fwd Packets": "Total Fwd Packet",
    "Total Forward Packets": "Total Fwd Packet",
    "Total Backward Packets": "Total Bwd packets",
    "Total Bwd Packets": "Total Bwd packets",

    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",

    "Avg Fwd Segment Size": "Fwd Segment Size Avg",
    "Avg Bwd Segment Size": "Bwd Segment Size Avg",

    "Init_Win_bytes_forward": "FWD Init Win Bytes",
    "Init_Win_bytes_backward": "Bwd Init Win Bytes",

    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",

    "Total TCP Flow Time": "Total Connection Flow Time",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAME_MAP)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_reference_schema(reference_schema_dir: Path) -> list[str]:
    csv_files = sorted(reference_schema_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No reference aligned CSV found in:\n{reference_schema_dir}\n"
            "Run Dataset 1 benign aligned extraction first."
        )

    reference_csv = csv_files[0]
    print(f"[*] Loading shared schema from: {reference_csv}")

    df = pd.read_csv(reference_csv, nrows=0)
    return [str(c).strip() for c in df.columns]


# ============================================================
# FILE / PROCESS UTILS
# ============================================================

def run_command(cmd: list[str], error_message: str) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(error_message)

    return result


def clean_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def docker_mount_path(path: Path) -> str:
    return path.resolve().as_posix()


def verify_docker_image(docker_image: str) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", docker_image],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Docker image not found: {docker_image}\n"
            "Run `docker images` and update DOCKER_IMAGE if needed."
        )


def verify_tool(tool_name: str) -> None:
    result = subprocess.run(
        [tool_name, "-h"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{tool_name} was not found.\n"
            "Install Wireshark or add this folder to PATH:\n"
            r"C:\Program Files\Wireshark"
        )


def convert_to_pcap(src: Path, dst: Path) -> None:
    print(f"[*] Converting to PCAP with editcap: {src.name}")
    cmd = ["editcap", "-F", "libpcap", str(src), str(dst)]
    run_command(cmd, f"editcap failed for: {src.name}")


def reorder_pcap_serial(src: Path, dst: Path) -> None:
    """Serial scripts' original reordercap behavior, including console output."""
    print(f"[*] Reordering packets by timestamp with reordercap: {src.name}")
    cmd = ["reordercap", str(src), str(dst)]
    run_command(cmd, f"reordercap failed for: {src.name}")


def reorder_pcap(src: Path, dst: Path) -> str:
    """Parallel scripts' original reordercap behavior: return tool message."""
    cmd = ["reordercap", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"reordercap failed for {src.name}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return (result.stdout or result.stderr or "").strip()


def prepare_cic_input_file(raw_file: Path, cic_input_dir: Path) -> Path:
    """Create one clean, timestamp-ordered libpcap file for serial extraction."""
    temp_unordered = cic_input_dir / f"{raw_file.stem}_unordered.pcap"
    final_ordered = cic_input_dir / f"{raw_file.stem}.pcap"

    if raw_file.suffix.lower() in {".pcap", ".pcapng"}:
        convert_to_pcap(raw_file, temp_unordered)
    else:
        raise ValueError(f"Unsupported file type: {raw_file}")

    reorder_pcap_serial(temp_unordered, final_ordered)

    if temp_unordered.exists():
        temp_unordered.unlink()

    return final_ordered


# ============================================================
# CICFLOWMETER EXECUTION
# ============================================================

def run_cicflowmeter_serial(run_root: Path, docker_image: str) -> None:
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_mount_path(run_root)}:/tmp/pcap",
        docker_image,
        "/tmp/pcap/input",
        "/tmp/pcap/output",
    ]

    print(f"[*] Running upgraded CICFlowMeter image: {docker_image}")
    run_command(cmd, "CICFlowMeter Docker run failed.")


def run_cicflowmeter_for_one_pcap(
    filtered_pcap_str: str,
    job_root_str: str,
    docker_image: str,
) -> tuple[str, bool, str]:
    """Run one CICFlowMeter Docker container for one filtered PCAP."""
    filtered_pcap = Path(filtered_pcap_str)
    job_root = Path(job_root_str)

    job_input = job_root / "input"
    job_output = job_root / "output"
    job_log = job_root / "cicflowmeter.log"

    try:
        clean_dir(job_input)
        clean_dir(job_output)

        dst_pcap = job_input / filtered_pcap.name
        shutil.copy2(filtered_pcap, dst_pcap)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{docker_mount_path(job_root)}:/tmp/pcap",
            docker_image,
            "/tmp/pcap/input",
            "/tmp/pcap/output",
        ]

        with job_log.open("w", encoding="utf-8", errors="ignore") as log:
            result = subprocess.run(cmd, stdout=log, stderr=log, text=True)

        if result.returncode != 0:
            return filtered_pcap.name, False, f"CICFlowMeter failed. See log: {job_log}"

        return filtered_pcap.name, True, str(job_output)

    except Exception as exc:
        return filtered_pcap.name, False, str(exc)


def run_cicflowmeter_parallel(
    cic_input_dir: Path,
    parallel_cic_root: Path,
    docker_image: str,
    max_workers: int,
) -> list[Path]:
    filtered_pcaps = sorted(cic_input_dir.glob("*.pcap"))

    if not filtered_pcaps:
        raise FileNotFoundError(f"No filtered PCAPs found in: {cic_input_dir}")

    clean_dir(parallel_cic_root)

    print("\n[Step 2] Running CICFlowMeter in parallel...")
    print(f"[*] Filtered PCAPs: {len(filtered_pcaps)}")
    print(f"[*] CICFlowMeter workers: {max_workers}")

    output_dirs: list[Path] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for filtered_pcap in filtered_pcaps:
            job_root = parallel_cic_root / filtered_pcap.stem
            futures.append(
                executor.submit(
                    run_cicflowmeter_for_one_pcap,
                    str(filtered_pcap),
                    str(job_root),
                    docker_image,
                )
            )

        for future in as_completed(futures):
            pcap_name, success, message = future.result()

            if success:
                output_dir = Path(message)
                output_dirs.append(output_dir)
                print(f"[OK] CICFlowMeter finished: {pcap_name}")
            else:
                print(f"[ERROR] CICFlowMeter failed: {pcap_name}")
                print(f"        {message}")

    if not output_dirs:
        raise RuntimeError("All CICFlowMeter jobs failed.")

    return output_dirs


# ============================================================
# PCAP PARSING HELPERS
# ============================================================

def inet_to_str(inet: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def open_pcap_reader(file_obj):
    try:
        return dpkt.pcap.Reader(file_obj)
    except ValueError:
        file_obj.seek(0)
        return dpkt.pcapng.Reader(file_obj)


def extract_ip_layer(buf: bytes, datalink: int, allow_linux_sll: bool = False):
    if datalink == dpkt.pcap.DLT_EN10MB:
        eth = dpkt.ethernet.Ethernet(buf)
        return eth.data

    if datalink == dpkt.pcap.DLT_RAW:
        return dpkt.ip.IP(buf)

    if (
        allow_linux_sll
        and hasattr(dpkt.pcap, "DLT_LINUX_SLL")
        and datalink == dpkt.pcap.DLT_LINUX_SLL
    ):
        try:
            return dpkt.sll.SLL(buf).data
        except Exception:
            return None

    return None
