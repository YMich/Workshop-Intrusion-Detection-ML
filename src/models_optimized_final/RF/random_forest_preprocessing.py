from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


class FoldMedianImputer:
    """Training-only median imputer used consistently by the final RF pipeline."""

    def __init__(self, name: str):
        self.name = str(name)
        self.columns_: list[str] | None = None
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "FoldMedianImputer":
        if X.empty:
            raise ValueError(f"{self.name}: cannot fit on an empty frame.")

        medians = X.median(axis=0, skipna=True)
        unusable = medians[medians.isna()].index.tolist()
        if unusable:
            raise ValueError(
                f"{self.name}: no usable training values for columns {unusable}"
            )

        self.columns_ = list(X.columns)
        self.medians_ = medians.astype(float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None or self.medians_ is None:
            raise RuntimeError(f"{self.name}: imputer must be fitted first.")
        if list(X.columns) != self.columns_:
            raise ValueError(f"{self.name}: feature schema/order differs from training.")

        transformed = X.fillna(self.medians_)
        if transformed.isna().any().any():
            bad = transformed.columns[transformed.isna().any()].tolist()
            raise ValueError(f"{self.name}: NaN remains after imputation: {bad}")
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FoldMedianImputer":
        return joblib.load(Path(path))
