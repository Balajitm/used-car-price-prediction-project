"""Clean the used-car dataset and write a modeling-ready copy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REFERENCE_YEAR = 2026
TARGET_COLUMN = "Selling_Price_Lakh"
IDENTIFIER_COLUMNS = ["Car_ID"]
LEAKAGE_CANDIDATES = ["Demand_Score", "Depreciation_Percent"]
CATEGORICAL_COLUMNS = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission",
    "Location",
    "Insurance_Valid",
    "Accident_History",
    "Service_History",
    "Dealer_or_Individual",
]
NUMERIC_COLUMNS = [
    "Year",
    "Car_Age",
    "KM_Driven",
    "KM_Per_Year",
    "Owner",
    "Present_Price_Lakh",
    "Condition_Score",
    "Engine_CC",
    "Mileage_KMPL",
    TARGET_COLUMN,
]


def normalise_category(value: object) -> object:
    if pd.isna(value):
        return value
    return re.sub(r"\s+", " ", str(value).strip())


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = data.copy()
    original_rows = len(cleaned)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].map(normalise_category)

    missing_service_history = int(cleaned["Service_History"].isna().sum())
    cleaned["Service_History"] = cleaned["Service_History"].fillna("Unknown")

    invalid_rows = (
        (cleaned["Year"] > REFERENCE_YEAR)
        | (cleaned["Year"] < 1900)
        | (cleaned["KM_Driven"] < 0)
        | (cleaned["Engine_CC"] <= 0)
        | (cleaned["Mileage_KMPL"] <= 0)
        | (cleaned[TARGET_COLUMN] <= 0)
    )
    invalid_row_count = int(invalid_rows.sum())
    cleaned = cleaned.loc[~invalid_rows].copy()

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    cleaned["Car_Age"] = REFERENCE_YEAR - cleaned["Year"]
    cleaned["KM_Per_Year"] = cleaned["KM_Driven"] / cleaned["Car_Age"].clip(lower=1)

    removed_columns = IDENTIFIER_COLUMNS + LEAKAGE_CANDIDATES
    cleaned = cleaned.drop(columns=removed_columns, errors="ignore")

    numeric_missing = {
        str(column): int(count)
        for column, count in cleaned.select_dtypes(include="number").isna().sum().items()
        if count
    }
    report = {
        "reference_year": REFERENCE_YEAR,
        "original_rows": original_rows,
        "final_rows": len(cleaned),
        "rows_removed_for_invalid_values": invalid_row_count,
        "duplicate_rows_removed": duplicate_rows,
        "service_history_values_filled": missing_service_history,
        "removed_columns": removed_columns,
        "numeric_missing_values_deferred_to_pipeline": numeric_missing,
        "remaining_columns": cleaned.columns.tolist(),
    }
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/used_car_dataset_100000.csv"),
        help="Path to the raw source CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/used_car_cleaned.csv"),
        help="Path for the cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data_cleaning.json"),
        help="Path for the cleaning report.",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    cleaned, report = clean_dataset(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {report['original_rows']:,} -> {report['final_rows']:,}")
    print(f"Service history values filled: {report['service_history_values_filled']:,}")
    print(f"Removed columns: {', '.join(report['removed_columns'])}")
    print(f"Cleaned data written to {args.output}")
    print(f"Cleaning report written to {args.report}")


if __name__ == "__main__":
    main()
