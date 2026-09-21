import pandas as pd

from cicflowmeter_common import SHARED_COLUMNS, normalize_columns
from pipeline_config import (
    CICIDS2017_FRIDAY_INPUT_CSV,
    CICIDS2017_FRIDAY_OUTPUT_CSV,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = CICIDS2017_FRIDAY_INPUT_CSV
OUTPUT_CSV = CICIDS2017_FRIDAY_OUTPUT_CSV

CHUNKSIZE = 100_000

# Columns that exist in the CICIDS2017 Friday CSV but are not part
# of the common final CICFlowMeter schema.
DROP_COLUMNS = {
    "id",
    "Attempted Category",
}


# ============================================================
# PROCESSING
# ============================================================

def filter_benign_https_tcp(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only benign TCP port-443 flows:

        Label == BENIGN
        Protocol == 6
        Src Port == 443 OR Dst Port == 443
    """
    required = {
        "Label",
        "Protocol",
        "Src Port",
        "Dst Port",
    }

    missing = required - set(chunk.columns)

    if missing:
        raise RuntimeError(
            f"Friday CSV is missing required filter columns: {sorted(missing)}"
        )

    benign_mask = (
        chunk["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("BENIGN")
    )

    protocol = pd.to_numeric(
        chunk["Protocol"],
        errors="coerce",
    )

    src_port = pd.to_numeric(
        chunk["Src Port"],
        errors="coerce",
    )

    dst_port = pd.to_numeric(
        chunk["Dst Port"],
        errors="coerce",
    )

    https_tcp_mask = (
        (protocol == 6)
        & (
            (src_port == 443)
            | (dst_port == 443)
        )
    )

    return chunk.loc[
        benign_mask & https_tcp_mask
    ].copy()


def normalize_friday_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one CICIDS2017 Friday chunk to the canonical shared schema.
    """
    chunk = chunk.copy()

    # Strip whitespace first so filtering works reliably.
    chunk.columns = [
        str(column).strip()
        for column in chunk.columns
    ]

    chunk = filter_benign_https_tcp(chunk)

    if chunk.empty:
        return chunk

    # Remove Friday/CICIDS-specific columns that are not part of
    # the shared final schema.
    existing_drop_columns = [
        column
        for column in DROP_COLUMNS
        if column in chunk.columns
    ]

    if existing_drop_columns:
        chunk = chunk.drop(
            columns=existing_drop_columns
        )

    # Uses the same normalization map as all CICFlowMeter scripts.
    # In particular:
    # Total TCP Flow Time -> Total Connection Flow Time
    chunk = normalize_columns(chunk)

    # Normalize label spelling to the rest of the extraction output.
    chunk["Label"] = "Benign"

    missing = [
        column
        for column in SHARED_COLUMNS
        if column not in chunk.columns
    ]

    extra = [
        column
        for column in chunk.columns
        if column not in SHARED_COLUMNS
    ]

    if missing or extra:
        message = [
            "CICIDS2017 Friday schema does not match the canonical schema."
        ]

        if missing:
            message.append(
                f"Missing columns: {missing}"
            )

        if extra:
            message.append(
                f"Extra columns: {extra}"
            )

        raise RuntimeError(
            "\n".join(message)
        )

    # Exact canonical names and exact canonical order.
    return chunk[SHARED_COLUMNS]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "CICIDS2017 Friday CSV not found:\n"
            f"{INPUT_CSV}\n\n"
            "Place friday.csv directly in the Dataset 1 raw benign folder:\n"
            f"{INPUT_CSV.parent}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    total_input_rows = 0
    total_benign_rows = 0
    total_benign_https_rows = 0
    wrote_header = False

    print("=" * 80)
    print("DATASET 1 - CICIDS2017 FRIDAY BENIGN HTTPS PREPARATION")
    print("=" * 80)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(
        "Filter: Label == BENIGN AND Protocol == 6 AND "
        "(Src Port == 443 OR Dst Port == 443)"
    )
    print(f"Canonical columns: {len(SHARED_COLUMNS)}")

    for chunk in pd.read_csv(
        INPUT_CSV,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        total_input_rows += len(chunk)

        stripped_columns = [
            str(column).strip()
            for column in chunk.columns
        ]
        chunk.columns = stripped_columns

        if "Label" not in chunk.columns:
            raise RuntimeError(
                "Friday CSV does not contain a Label column."
            )

        benign_mask = (
            chunk["Label"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("BENIGN")
        )

        total_benign_rows += int(
            benign_mask.sum()
        )

        normalized = normalize_friday_chunk(
            chunk
        )

        if normalized.empty:
            continue

        normalized.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=not wrote_header,
        )

        wrote_header = True
        total_benign_https_rows += len(
            normalized
        )

    if not wrote_header:
        raise RuntimeError(
            "No BENIGN TCP/443 rows were found in friday.csv."
        )

    final_columns = list(
        pd.read_csv(
            OUTPUT_CSV,
            nrows=0,
        ).columns
    )

    if final_columns != SHARED_COLUMNS:
        raise RuntimeError(
            "Final Friday output header does not exactly match "
            "the canonical shared schema."
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Input rows:               {total_input_rows:,}"
    )
    print(
        f"All BENIGN rows:          {total_benign_rows:,}"
    )
    print(
        f"BENIGN TCP/443 rows kept: {total_benign_https_rows:,}"
    )
    print(
        f"Final columns:            {len(final_columns)}"
    )
    print(
        f"Saved to:                 {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
