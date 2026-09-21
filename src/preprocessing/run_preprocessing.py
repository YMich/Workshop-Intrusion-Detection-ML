"""Standalone Milestone-2 preprocessing snapshot runner.

This script intentionally fits one ``DatasetPreprocessor`` to each complete
Dataset 1/2/3 CSV in order to reproduce the earlier Milestone-2 normalized
snapshots under ``data/preprocessed``.

It is NOT the preprocessing entry point used by final held-out model
evaluation. Final model pipelines create the train/validation/test split first
and fit ``DatasetPreprocessor`` only on TRAIN data (or benign TRAIN for the
one-class Autoencoder and Isolation Forest).
"""

import pandas as pd

try:
    from .preprocessing import preprocess_all_datasets
    from .preprocessing_config import (
        DATASET1_INGESTED_CSV,
        DATASET1_PREPROCESSED_CSV,
        DATASET1_PREPROCESSOR_ARTIFACT,
        DATASET2_INGESTED_CSV,
        DATASET2_PREPROCESSED_CSV,
        DATASET2_PREPROCESSOR_ARTIFACT,
        DATASET3_INGESTED_CSV,
        DATASET3_PREPROCESSED_CSV,
        DATASET3_PREPROCESSOR_ARTIFACT,
    )
except ImportError:
    from preprocessing import preprocess_all_datasets
    from preprocessing_config import (
        DATASET1_INGESTED_CSV,
        DATASET1_PREPROCESSED_CSV,
        DATASET1_PREPROCESSOR_ARTIFACT,
        DATASET2_INGESTED_CSV,
        DATASET2_PREPROCESSED_CSV,
        DATASET2_PREPROCESSOR_ARTIFACT,
        DATASET3_INGESTED_CSV,
        DATASET3_PREPROCESSED_CSV,
        DATASET3_PREPROCESSOR_ARTIFACT,
    )


DATASET_INPUTS = {
    "dataset1": (
        DATASET1_INGESTED_CSV
    ),
    "dataset2": (
        DATASET2_INGESTED_CSV
    ),
    "dataset3": (
        DATASET3_INGESTED_CSV
    ),
}


DATASET_OUTPUTS = {
    "dataset1": (
        DATASET1_PREPROCESSED_CSV
    ),
    "dataset2": (
        DATASET2_PREPROCESSED_CSV
    ),
    "dataset3": (
        DATASET3_PREPROCESSED_CSV
    ),
}


PREPROCESSOR_ARTIFACTS = {
    "dataset1": (
        DATASET1_PREPROCESSOR_ARTIFACT
    ),
    "dataset2": (
        DATASET2_PREPROCESSOR_ARTIFACT
    ),
    "dataset3": (
        DATASET3_PREPROCESSOR_ARTIFACT
    ),
}


def load_ingested_datasets():
    datasets = {}

    for (
        dataset_name,
        input_path,
    ) in DATASET_INPUTS.items():
        if not input_path.exists():
            raise FileNotFoundError(
                f"{dataset_name}: "
                f"ingested CSV not found:\n"
                f"{input_path}"
            )

        df = pd.read_csv(
            input_path,
            low_memory=False,
        )

        datasets[
            dataset_name
        ] = df

        print(
            f"Loaded {dataset_name}: "
            f"{input_path} "
            f"| shape={df.shape}"
        )

    return datasets


def save_preprocessed_datasets(
    datasets,
):
    for (
        dataset_name,
        df,
    ) in datasets.items():
        output_path = (
            DATASET_OUTPUTS[
                dataset_name
            ]
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
            date_format=(
                "%Y-%m-%d "
                "%H:%M:%S.%f"
            ),
        )

        print(
            f"Saved {dataset_name}: "
            f"{output_path} "
            f"| shape={df.shape}"
        )


def save_preprocessors(
    preprocessors,
):
    for (
        dataset_name,
        preprocessor,
    ) in preprocessors.items():
        artifact_path = (
            PREPROCESSOR_ARTIFACTS[
                dataset_name
            ]
        )

        preprocessor.save(
            artifact_path
        )

        print(
            f"Saved {dataset_name} "
            f"preprocessing artifact: "
            f"{artifact_path}"
        )


def main():
    print("=" * 80)
    print(
        "STEP 2 - "
        "PREPROCESSING & NORMALIZATION"
    )
    print("=" * 80)
    print(
        "\nNOTE: This standalone runner reproduces the Milestone-2 "
        "full-dataset preprocessing snapshots."
    )
    print(
        "It is NOT used by the final held-out model evaluation pipelines."
    )
    print(
        "Final models fit DatasetPreprocessor on TRAIN only "
        "(or benign TRAIN for AE/IF).\n"
    )

    datasets = (
        load_ingested_datasets()
    )

    (
        preprocessed,
        preprocessors,
    ) = preprocess_all_datasets(
        datasets
    )

    save_preprocessed_datasets(
        preprocessed
    )

    save_preprocessors(
        preprocessors
    )

    print("\n" + "=" * 80)
    print(
        "PREPROCESSING & "
        "NORMALIZATION COMPLETE"
    )
    print("=" * 80)

    for (
        dataset_name,
        preprocessor,
    ) in preprocessors.items():
        print(
            f"{dataset_name}: "
            f"{len(preprocessor.feature_columns_)} "
            f"predictive features normalized "
            "using the complete dataset's own "
            "median + RobustScaler for this standalone snapshot."
        )

    print(
        "\nNo feature selection, "
        "identifier removal, "
        "sequence construction, "
        "model training, or evaluation "
        "was performed."
    )


if __name__ == "__main__":
    main()
