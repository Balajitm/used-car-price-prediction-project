"""Create leakage-safe features from the cleaned used-car dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Selling_Price_Lakh"


def engineer_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    engineered = data.copy()
    added_features: list[str] = []

    engineered["Car_Age_Squared"] = engineered["Car_Age"] ** 2
    added_features.append("Car_Age_Squared")

    engineered["Log_KM_Driven"] = np.log1p(engineered["KM_Driven"].clip(lower=0))
    added_features.append("Log_KM_Driven")

    engineered["Log_Present_Price_Lakh"] = np.log1p(
        engineered["Present_Price_Lakh"].clip(lower=0)
    )
    added_features.append("Log_Present_Price_Lakh")

    engineered["Condition_Age_Interaction"] = (
        engineered["Condition_Score"] / engineered["Car_Age"].clip(lower=1)
    )
    added_features.append("Condition_Age_Interaction")

    engineered["Recent_Vehicle"] = (engineered["Car_Age"] <= 3).astype(int)
    added_features.append("Recent_Vehicle")

    report = {
        "input_rows": int(len(data)),
        "output_rows": int(len(engineered)),
        "target_unchanged": bool(engineered[TARGET_COLUMN].equals(data[TARGET_COLUMN])),
        "added_features": added_features,
        "target_dependent_features_added": [],
        "deferred_features": {
            "brand_target_encoding": "Must be fitted on training folds only.",
            "location_target_encoding": "Must be fitted on training folds only.",
        },
    }
    return engineered, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/used_car_cleaned.csv"),
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/used_car_features.csv"),
        help="Path for the feature-enhanced CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/feature_engineering.json"),
        help="Path for the feature engineering report.",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    engineered, report = engineer_features(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {report['input_rows']:,} -> {report['output_rows']:,}")
    print(f"Added features: {', '.join(report['added_features'])}")
    print(f"Feature-enhanced data written to {args.output}")
    print(f"Feature report written to {args.report}")


if __name__ == "__main__":
    main()
