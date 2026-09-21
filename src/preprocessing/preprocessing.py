from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    # Package-style import, e.g. ``from src.preprocessing import ...``.
    from .preprocessing_config import (
        LABEL_COL,
        METADATA_COLUMNS,
        PASSTHROUGH_COLUMNS,
        ROBUST_SCALER_QUANTILE_RANGE,
        ROBUST_SCALER_UNIT_VARIANCE,
        ROBUST_SCALER_WITH_CENTERING,
        ROBUST_SCALER_WITH_SCALING,
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
    )
except ImportError:
    # Direct-module import used by the final model folders after adding
    # ``src/preprocessing`` to ``sys.path``.
    from preprocessing_config import (
        LABEL_COL,
        METADATA_COLUMNS,
        PASSTHROUGH_COLUMNS,
        ROBUST_SCALER_QUANTILE_RANGE,
        ROBUST_SCALER_UNIT_VARIANCE,
        ROBUST_SCALER_WITH_CENTERING,
        ROBUST_SCALER_WITH_SCALING,
        SOURCE_FILE_COL,
        TIMESTAMP_COL,
    )


# ============================================================
# NON-FINITE VALUES
# ============================================================

NONFINITE_TEXT_VALUES = [
    "Infinity",
    "-Infinity",
    "+Infinity",
    "inf",
    "-inf",
    "+inf",
    "INF",
    "-INF",
    "+INF",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def signed_log1p(values):
    """
    Apply the report-defined signed log1p transformation:

        sign(x) * ln(1 + |x|)

    This preserves the sign while compressing large magnitudes.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        np.sign(values)
        * np.log1p(
            np.abs(values)
        )
    )


def clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Defensive header cleanup for CSV snapshots.
    """
    df = df.copy()

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in df.columns
    ]

    duplicated = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated:
        raise ValueError(
            "Duplicate columns after header cleanup: "
            f"{duplicated}"
        )

    return df


def parse_timestamp_if_needed(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore Timestamp to datetime when an ingested CSV snapshot is loaded.
    """
    df = df.copy()

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing required metadata column: "
            f"{TIMESTAMP_COL}"
        )

    if pd.api.types.is_datetime64_any_dtype(
        df[TIMESTAMP_COL]
    ):
        return df

    try:
        parsed = pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
        )

    invalid_mask = parsed.isna()

    if invalid_mask.any():
        examples = (
            df.loc[
                invalid_mask,
                TIMESTAMP_COL,
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Could not parse "
            f"{int(invalid_mask.sum()):,} "
            f"Timestamp value(s). "
            f"Examples: {examples}"
        )

    df[TIMESTAMP_COL] = parsed

    return df


def validate_passthrough_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that metadata/identifier fields required by the later LSTM
    engineering stage are still present.
    """
    missing = [
        column
        for column in PASSTHROUGH_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required metadata/label columns: "
            f"{missing}"
        )


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    At Step 2 every non-metadata/non-label field is still a candidate
    predictive feature.

    No feature selection is performed here.
    """
    return [
        column
        for column in df.columns
        if column not in PASSTHROUGH_COLUMNS
    ]


def replace_nonfinite_with_nan(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Replace CICFlowMeter inf/-inf artifacts with NaN in predictive features.
    """
    df = df.copy()

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [
                np.inf,
                -np.inf,
                *NONFINITE_TEXT_VALUES,
            ],
            np.nan,
        )
    )

    return df


def convert_features_to_numeric(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert candidate predictive features to numeric.

    Any invalid numeric value becomes NaN and is later filled using the
    median learned from the dataframe supplied to ``DatasetPreprocessor.fit``.
    """
    df = df.copy()

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def clean_before_normalization(
    df: pd.DataFrame,
    expected_feature_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply deterministic preprocessing before median imputation and scaling.
    """
    df = clean_column_names(
        df
    )

    df = parse_timestamp_if_needed(
        df
    )

    validate_passthrough_columns(
        df
    )

    feature_columns = (
        get_feature_columns(
            df
        )
    )

    if expected_feature_columns is not None:
        if feature_columns != expected_feature_columns:
            missing = [
                column
                for column in expected_feature_columns
                if column not in feature_columns
            ]

            extra = [
                column
                for column in feature_columns
                if column not in expected_feature_columns
            ]

            order_difference = (
                not missing
                and not extra
                and feature_columns
                != expected_feature_columns
            )

            message = [
                "Predictive feature schema mismatch."
            ]

            if missing:
                message.append(
                    f"Missing features: {missing}"
                )

            if extra:
                message.append(
                    f"Extra features: {extra}"
                )

            if order_difference:
                message.append(
                    "Feature order is different."
                )

            raise ValueError(
                "\n".join(message)
            )

    df = replace_nonfinite_with_nan(
        df,
        feature_columns,
    )

    df = convert_features_to_numeric(
        df,
        feature_columns,
    )

    return (
        df,
        feature_columns,
    )


# ============================================================
# DATASET PREPROCESSOR
# ============================================================

class DatasetPreprocessor:
    """
    Fit preprocessing parameters on a reference dataframe and apply them later.

    Numerical pipeline:

        inf / -inf
            -> NaN
            -> fitted-reference median imputation
            -> signed log1p
            -> fitted-reference RobustScaler

    The fit scope is controlled by the caller.

    Final supervised models fit this object on TRAIN only. The one-class
    Autoencoder and Isolation Forest fit it on benign TRAIN only. The optional
    standalone Milestone-2 runner intentionally fits one instance to each
    complete dataset to reproduce the earlier full-dataset preprocessing
    snapshots.

    Metadata, identifiers, Timestamp, SourceFile, and Label are retained
    unchanged.
    """

    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = (
            str(dataset_name)
        )

        self.feature_columns_: Optional[
            list[str]
        ] = None

        self.raw_columns_: Optional[
            list[str]
        ] = None

        self.medians_: Optional[
            pd.Series
        ] = None

        self.scaler_: Optional[
            RobustScaler
        ] = None

        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "DatasetPreprocessor":
        """
        Learn median-imputation values and RobustScaler parameters from ``df``.

        The caller determines the fit scope. Final evaluation pipelines pass
        only TRAIN data (or benign TRAIN for one-class models).
        """
        (
            cleaned,
            feature_columns,
        ) = clean_before_normalization(
            df
        )

        if not feature_columns:
            raise ValueError(
                f"{self.dataset_name}: "
                "no predictive feature columns found."
            )

        self.feature_columns_ = list(
            feature_columns
        )

        self.raw_columns_ = list(
            cleaned.columns
        )

        medians = (
            cleaned[
                self.feature_columns_
            ]
            .median(
                axis=0,
                skipna=True,
            )
        )

        all_nan_columns = (
            medians[
                medians.isna()
            ]
            .index
            .tolist()
        )

        if all_nan_columns:
            raise ValueError(
                f"{self.dataset_name}: "
                "median imputation is undefined because "
                "these feature columns contain no valid numeric values: "
                f"{all_nan_columns}"
            )

        self.medians_ = (
            medians.astype(
                np.float64
            )
        )

        imputed_features = (
            cleaned[
                self.feature_columns_
            ]
            .fillna(
                self.medians_
            )
            .astype(
                np.float64
            )
        )

        logged_features = (
            signed_log1p(
                imputed_features.to_numpy()
            )
        )

        self.scaler_ = RobustScaler(
            with_centering=(
                ROBUST_SCALER_WITH_CENTERING
            ),
            with_scaling=(
                ROBUST_SCALER_WITH_SCALING
            ),
            quantile_range=(
                ROBUST_SCALER_QUANTILE_RANGE
            ),
            unit_variance=(
                ROBUST_SCALER_UNIT_VARIANCE
            ),
        )

        self.scaler_.fit(
            logged_features
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the already-fitted preprocessing parameters without refitting.

        This is the operation used to transform validation, test, and transfer
        data after preprocessing has been fitted on the permitted training
        reference data.
        """
        self._check_is_fitted()

        (
            cleaned,
            _,
        ) = clean_before_normalization(
            df,
            expected_feature_columns=(
                self.feature_columns_
            ),
        )

        if list(
            cleaned.columns
        ) != self.raw_columns_:
            raise ValueError(
                f"{self.dataset_name}: "
                "input columns/order differ from the fitted schema."
            )

        features = (
            cleaned[
                self.feature_columns_
            ]
            .fillna(
                self.medians_
            )
            .astype(
                np.float64
            )
        )

        remaining_nan = (
            features.isna().any()
        )

        if remaining_nan.any():
            columns = (
                remaining_nan[
                    remaining_nan
                ]
                .index
                .tolist()
            )

            raise ValueError(
                f"{self.dataset_name}: "
                f"NaN remains after median imputation in: {columns}"
            )

        logged = signed_log1p(
            features.to_numpy()
        )

        scaled = self.scaler_.transform(
            logged
        )

        result = cleaned.copy()

        scaled_df = pd.DataFrame(
            scaled,
            index=result.index,
            columns=(
                self.feature_columns_
            ),
            dtype=np.float64,
        )

        for column in self.feature_columns_:
            result[column] = (
                scaled_df[column]
            )

        # Keep Label as an integer binary target.
        label_numeric = pd.to_numeric(
            result[LABEL_COL],
            errors="coerce",
        )

        if label_numeric.isna().any():
            raise ValueError(
                f"{self.dataset_name}: "
                "Label contains invalid non-numeric value(s)."
            )

        labels = set(
            label_numeric
            .astype(int)
            .unique()
            .tolist()
        )

        invalid_labels = (
            labels - {0, 1}
        )

        if invalid_labels:
            raise ValueError(
                f"{self.dataset_name}: "
                "Label must contain only 0/1. "
                f"Found: {sorted(invalid_labels)}"
            )

        result[LABEL_COL] = (
            label_numeric.astype(int)
        )

        # Explicitly preserve metadata.
        for column in METADATA_COLUMNS:
            result[column] = (
                cleaned[column]
            )

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit preprocessing on ``df`` and immediately transform the same frame.

        The standalone Milestone-2 runner uses this convenience method on a
        complete dataset. Final model pipelines instead control the fit scope
        explicitly before transforming validation/test data.
        """
        self.fit(
            df
        )

        return self.transform(
            df
        )

    def save(
        self,
        path: Path | str,
    ) -> None:
        self._check_is_fitted()

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> "DatasetPreprocessor":
        obj = joblib.load(
            Path(path)
        )

        if not isinstance(
            obj,
            cls,
        ):
            raise TypeError(
                "Loaded object is not a DatasetPreprocessor."
            )

        obj._check_is_fitted()

        return obj

    def summary(
        self,
    ) -> dict:
        self._check_is_fitted()

        return {
            "dataset_name": (
                self.dataset_name
            ),
            "n_features": len(
                self.feature_columns_
            ),
            "feature_columns": list(
                self.feature_columns_
            ),
            "median_imputation": (
                "median learned from fit reference"
            ),
            "transform": (
                "signed log1p"
            ),
            "scaler": (
                "RobustScaler learned from fit reference"
            ),
            "quantile_range": tuple(
                self.scaler_.quantile_range
            ),
            "metadata_columns": list(
                METADATA_COLUMNS
            ),
        }

    def _check_is_fitted(
        self,
    ) -> None:
        if (
            not self.is_fitted_
            or self.feature_columns_ is None
            or self.raw_columns_ is None
            or self.medians_ is None
            or self.scaler_ is None
        ):
            raise RuntimeError(
                f"{self.dataset_name}: "
                "DatasetPreprocessor has not been fitted."
            )


# ============================================================
# ALL-DATASET CONVENIENCE FUNCTION
# ============================================================

def preprocess_all_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, DatasetPreprocessor],
]:
    """
    Reproduce the standalone Milestone-2 preprocessing snapshots.

    Each complete dataset is fitted and transformed independently, so no
    median or scaler parameters are shared across datasets.

    IMPORTANT:
        This convenience function is not used for final held-out model
        evaluation. Final model pipelines establish their split first and fit
        ``DatasetPreprocessor`` only on TRAIN data (or benign TRAIN for the
        one-class models).
    """
    if not datasets:
        raise ValueError(
            "No datasets were provided."
        )

    processed = {}
    preprocessors = {}

    expected_feature_columns = None

    for (
        dataset_name,
        df,
    ) in datasets.items():
        preprocessor = (
            DatasetPreprocessor(
                dataset_name
            )
        )

        processed_df = (
            preprocessor.fit_transform(
                df
            )
        )

        if expected_feature_columns is None:
            expected_feature_columns = (
                list(
                    preprocessor.feature_columns_
                )
            )
        else:
            if (
                list(
                    preprocessor.feature_columns_
                )
                != expected_feature_columns
            ):
                raise ValueError(
                    f"{dataset_name}: predictive feature schema "
                    "differs from the other datasets."
                )

        processed[
            dataset_name
        ] = processed_df

        preprocessors[
            dataset_name
        ] = preprocessor

    return (
        processed,
        preprocessors,
    )
