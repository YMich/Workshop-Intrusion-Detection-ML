import json

import pandas as pd

from feature_config import (
    DATASET1_FEATURES_CSV,
    DATASET1_PREPROCESSED_CSV,
    DATASET2_FEATURES_CSV,
    DATASET2_PREPROCESSED_CSV,
    DATASET3_FEATURES_CSV,
    DATASET3_PREPROCESSED_CSV,
    FEATURE_ARTIFACT_ROOT,
)

from feature_selection import (
    select_and_engineer_all_datasets,
)


DATASET_INPUTS = {
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


DATASET_OUTPUTS = {
    "dataset1": (
        DATASET1_FEATURES_CSV
    ),
    "dataset2": (
        DATASET2_FEATURES_CSV
    ),
    "dataset3": (
        DATASET3_FEATURES_CSV
    ),
}


def load_preprocessed_datasets():
    datasets = {}

    for (
        dataset_name,
        input_path,
    ) in DATASET_INPUTS.items():
        if not input_path.exists():
            raise FileNotFoundError(
                f"{dataset_name}: "
                f"preprocessed CSV not found:\n"
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


def save_outputs(
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


def save_feature_artifacts(
    selected_features,
    drop_manifest,
):
    FEATURE_ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_csv = (
        FEATURE_ARTIFACT_ROOT
        / "selected_features.csv"
    )

    selected_json = (
        FEATURE_ARTIFACT_ROOT
        / "selected_features.json"
    )

    manifest_csv = (
        FEATURE_ARTIFACT_ROOT
        / "feature_drop_manifest.csv"
    )

    pd.DataFrame({
        "Feature": selected_features
    }).to_csv(
        selected_csv,
        index=False,
    )

    with selected_json.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "feature_count": len(
                    selected_features
                ),
                "features": (
                    selected_features
                ),
            },
            handle,
            indent=2,
        )

    drop_manifest.to_csv(
        manifest_csv,
        index=False,
    )

    print(
        f"Saved selected feature list: "
        f"{selected_csv}"
    )

    print(
        f"Saved feature manifest: "
        f"{manifest_csv}"
    )


def main():
    print("=" * 80)
    print(
        "STEP 3 - "
        "FEATURE SELECTION & ENGINEERING"
    )
    print("=" * 80)

    datasets = (
        load_preprocessed_datasets()
    )

    (
        feature_selected,
        selected_features,
        drop_manifest,
    ) = select_and_engineer_all_datasets(
        datasets
    )

    save_outputs(
        feature_selected
    )

    save_feature_artifacts(
        selected_features,
        drop_manifest,
    )

    print("\n" + "=" * 80)
    print(
        "FEATURE SELECTION & "
        "ENGINEERING COMPLETE"
    )
    print("=" * 80)

    print(
        f"Final predictive feature count: "
        f"{len(selected_features)}"
    )

    print(
        "\nEach output contains:"
        "\n  retained metadata/identifiers"
        "\n  + 57 selected numeric features"
        "\n  + Label"
    )

    print(
        "\nNo model-specific representation, "
        "sequence construction, model training, "
        "or evaluation was performed."
    )


if __name__ == "__main__":
    main()
