"""Build and fit the leakage-safe preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Selling_Price_Lakh"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/used_car_features.csv"),
        help="Path to the engineered feature CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/preprocessor.joblib"),
        help="Path for the fitted preprocessing object.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("reports/preprocessing.json"),
        help="Path for preprocessing metadata.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column not found: {TARGET_COLUMN}")

    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, args.model_output)

    metadata = {
        "input": str(args.input),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "transformed_train_shape": list(x_train_transformed.shape),
        "transformed_test_shape": list(x_test_transformed.shape),
        "train_target_mean_lakh": float(y_train.mean()),
        "test_target_mean_lakh": float(y_test.mean()),
        "unknown_categories": "Ignored by OneHotEncoder(handle_unknown='ignore').",
        "fit_scope": "Preprocessor fitted on training rows only.",
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Train rows: {metadata['train_rows']:,}")
    print(f"Test rows: {metadata['test_rows']:,}")
    print(f"Transformed train shape: {tuple(x_train_transformed.shape)}")
    print(f"Preprocessor written to {args.model_output}")
    print(f"Metadata written to {args.metadata_output}")


if __name__ == "__main__":
    main()
