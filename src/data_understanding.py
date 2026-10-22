"""Load and profile the used-car dataset without modifying source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Selling_Price_Lakh"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the source CSV and fail clearly when the target is unavailable."""
    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column not found: {TARGET_COLUMN}")
    return data


def build_profile(data: pd.DataFrame, data_path: Path) -> dict:
    """Create a serialisable profile for data understanding and review."""
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    target = data[TARGET_COLUMN]

    numeric_summary = data[numeric_columns].describe().transpose()
    numeric_summary = numeric_summary[["mean", "std", "min", "25%", "50%", "75%", "max"]]

    return {
        "source": str(data_path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_order": data.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {
            str(column): int(count)
            for column, count in data.isna().sum().items()
            if count
        },
        "unique_values": {
            str(column): int(data[column].nunique(dropna=True))
            for column in data.columns
        },
        "categorical_examples": {
            str(column): sorted(data[column].dropna().astype(str).unique().tolist())[:20]
            for column in categorical_columns
        },
        "numeric_summary": {
            str(column): {
                str(statistic): float(value)
                for statistic, value in values.items()
            }
            for column, values in numeric_summary.iterrows()
        },
        "target_summary": {
            "name": TARGET_COLUMN,
            "mean_lakh": float(target.mean()),
            "median_lakh": float(target.median()),
            "std_lakh": float(target.std()),
            "min_lakh": float(target.min()),
            "max_lakh": float(target.max()),
            "skewness": float(target.skew()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/used_car_dataset_100000.csv"),
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_understanding.json"),
        help="Path for the JSON profile.",
    )
    args = parser.parse_args()

    data = load_dataset(args.input)
    profile = build_profile(data, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Loaded {profile['rows']:,} rows and {profile['columns']} columns.")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Target median: ₹{profile['target_summary']['median_lakh']:.2f} lakh")
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()
