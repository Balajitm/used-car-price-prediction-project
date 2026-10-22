"""Generate reproducible exploratory analysis artifacts for used-car prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_COLUMN = "Selling_Price_Lakh"


def load_cleaned_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column not found: {TARGET_COLUMN}")
    return data


def save_plot(plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_plots(data: pd.DataFrame, output_dir: Path) -> list[str]:
    sns.set_theme(style="whitegrid", palette="deep")
    created_plots: list[str] = []

    plt.figure(figsize=(9, 5))
    sns.histplot(data[TARGET_COLUMN], bins=40, kde=True)
    plt.title("Selling Price Distribution")
    plt.xlabel("Selling price (lakh)")
    path = output_dir / "01_price_distribution.png"
    save_plot(path)
    created_plots.append(str(path))

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=data.sample(min(10000, len(data)), random_state=42), x="Car_Age", y=TARGET_COLUMN, alpha=0.25)
    sns.regplot(data=data, x="Car_Age", y=TARGET_COLUMN, scatter=False, color="black")
    plt.title("Vehicle Age and Selling Price")
    plt.xlabel("Car age (years)")
    plt.ylabel("Selling price (lakh)")
    path = output_dir / "02_age_vs_price.png"
    save_plot(path)
    created_plots.append(str(path))

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=data.sample(min(10000, len(data)), random_state=42), x="KM_Driven", y=TARGET_COLUMN, alpha=0.25)
    plt.title("Kilometres Driven and Selling Price")
    plt.xlabel("Kilometres driven")
    plt.ylabel("Selling price (lakh)")
    path = output_dir / "03_km_vs_price.png"
    save_plot(path)
    created_plots.append(str(path))

    for column, filename, title in [
        ("Brand", "04_brand_price.png", "Selling Price by Brand"),
        ("Fuel_Type", "05_fuel_price.png", "Selling Price by Fuel Type"),
        ("Transmission", "06_transmission_price.png", "Selling Price by Transmission"),
        ("Owner", "07_owner_price.png", "Selling Price by Previous Owner Count"),
        ("Location", "08_location_price.png", "Selling Price by Location"),
    ]:
        order = data.groupby(column)[TARGET_COLUMN].median().sort_values(ascending=False).index
        plt.figure(figsize=(11, 5))
        sns.boxplot(data=data, x=column, y=TARGET_COLUMN, order=order, showfliers=False)
        plt.title(title)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Selling price (lakh)")
        path = output_dir / filename
        save_plot(path)
        created_plots.append(str(path))

    numeric_columns = data.select_dtypes(include="number").columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[numeric_columns].corr(), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Numeric Feature Correlation Matrix")
    path = output_dir / "09_correlation_matrix.png"
    save_plot(path)
    created_plots.append(str(path))

    return created_plots


def build_findings(data: pd.DataFrame, plot_paths: list[str]) -> dict:
    brand_medians = data.groupby("Brand")[TARGET_COLUMN].median().sort_values(ascending=False)
    fuel_medians = data.groupby("Fuel_Type")[TARGET_COLUMN].median().sort_values(ascending=False)
    location_medians = data.groupby("Location")[TARGET_COLUMN].median().sort_values(ascending=False)
    correlation = data.select_dtypes(include="number").corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    correlation = correlation.reindex(correlation.abs().sort_values(ascending=False).index)

    return {
        "rows_analysed": int(len(data)),
        "plots": plot_paths,
        "target": {
            "mean_lakh": float(data[TARGET_COLUMN].mean()),
            "median_lakh": float(data[TARGET_COLUMN].median()),
            "q1_lakh": float(data[TARGET_COLUMN].quantile(0.25)),
            "q3_lakh": float(data[TARGET_COLUMN].quantile(0.75)),
        },
        "group_median_price_lakh": {
            "brand_highest": {"category": str(brand_medians.index[0]), "value": float(brand_medians.iloc[0])},
            "brand_lowest": {"category": str(brand_medians.index[-1]), "value": float(brand_medians.iloc[-1])},
            "fuel_highest": {"category": str(fuel_medians.index[0]), "value": float(fuel_medians.iloc[0])},
            "location_highest": {"category": str(location_medians.index[0]), "value": float(location_medians.iloc[0])},
        },
        "target_correlations": {str(column): float(value) for column, value in correlation.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/used_car_cleaned.csv"),
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for EDA plots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/eda_findings.json"),
        help="Path for the EDA findings JSON.",
    )
    args = parser.parse_args()

    data = load_cleaned_data(args.input)
    plot_paths = create_plots(data, args.output_dir)
    findings = build_findings(data, plot_paths)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print(f"Analysed {findings['rows_analysed']:,} rows.")
    print(f"Created {len(plot_paths)} plots in {args.output_dir}.")
    print(f"EDA findings written to {args.report}.")


if __name__ == "__main__":
    main()
