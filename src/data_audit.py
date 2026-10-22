"""Audit the supplied used-car dataset without modifying source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "Car_ID",
    "Brand",
    "Model",
    "Year",
    "Car_Age",
    "KM_Driven",
    "KM_Per_Year",
    "Fuel_Type",
    "Transmission",
    "Owner",
    "Location",
    "Present_Price_Lakh",
    "Insurance_Valid",
    "Accident_History",
    "Service_History",
    "Condition_Score",
    "Engine_CC",
    "Mileage_KMPL",
    "Dealer_or_Individual",
    "Demand_Score",
    "Depreciation_Percent",
    "Selling_Price_Lakh",
}

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
    "Demand_Score",
    "Depreciation_Percent",
    "Selling_Price_Lakh",
]


def build_audit_report(data_path: Path) -> dict:
    data = pd.read_csv(data_path)
    numeric_data = data[NUMERIC_COLUMNS]
    derived_age_error = (data["Car_Age"] - (2026 - data["Year"])).abs()
    expected_km_per_year = data["KM_Driven"] / data["Car_Age"].replace(0, pd.NA)
    derived_km_error = (data["KM_Per_Year"] - expected_km_per_year).abs()

    report = {
        "file": str(data_path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_names": list(data.columns),
        "missing_values": {
            str(key): int(value)
            for key, value in data.isna().sum().items()
            if value
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_car_ids": int(data["Car_ID"].duplicated().sum()),
        "unexpected_columns": sorted(set(data.columns) - EXPECTED_COLUMNS),
        "missing_expected_columns": sorted(EXPECTED_COLUMNS - set(data.columns)),
        "dtypes": {str(key): str(value) for key, value in data.dtypes.items()},
        "categorical_cardinality": {
            str(column): int(data[column].nunique(dropna=True))
            for column in data.select_dtypes(include="object").columns
        },
        "numeric_ranges": {
            str(column): {
                "min": float(numeric_data[column].min()),
                "max": float(numeric_data[column].max()),
            }
            for column in NUMERIC_COLUMNS
        },
        "invalid_value_counts": {
            "future_year": int((data["Year"] > 2026).sum()),
            "negative_kilometers": int((data["KM_Driven"] < 0).sum()),
            "non_positive_engine_cc": int((data["Engine_CC"] <= 0).sum()),
            "non_positive_mileage": int((data["Mileage_KMPL"] <= 0).sum()),
            "non_positive_selling_price": int((data["Selling_Price_Lakh"] <= 0).sum()),
        },
        "derived_feature_checks": {
            "car_age_mismatch_using_2026_reference": int((derived_age_error > 0).sum()),
            "km_per_year_mismatch_over_1_km": int((derived_km_error > 1).sum()),
        },
        "categorical_values": {
            str(column): sorted(data[column].dropna().astype(str).unique().tolist())
            for column in data.select_dtypes(include="object").columns
        },
    }
    return report


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
        default=Path("reports/data_audit.json"),
        help="Path for the JSON audit report.",
    )
    args = parser.parse_args()

    report = build_audit_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
