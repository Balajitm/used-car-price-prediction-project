"""Evaluate the selected model and calculate permutation-based explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_engineering import TARGET_COLUMN
from preprocessing import RANDOM_STATE, TEST_SIZE


def make_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    errors = actual.to_numpy() - predicted
    non_zero = actual != 0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": float(
            np.mean(np.abs(errors[non_zero] / actual.to_numpy()[non_zero])) * 100
        ),
        "mean_error_lakh": float(np.mean(errors)),
        "median_absolute_error_lakh": float(np.median(np.abs(errors))),
    }


def save_plots(
    actual: pd.Series,
    predicted: np.ndarray,
    residuals: np.ndarray,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: list[str] = []

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.25)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    plt.plot(limits, limits, color="black", linestyle="--", label="Perfect prediction")
    plt.xlabel("Actual selling price (lakh)")
    plt.ylabel("Predicted selling price (lakh)")
    plt.title("Actual versus Predicted Price")
    plt.legend()
    path = output_dir / "01_actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.25)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted selling price (lakh)")
    plt.ylabel("Residual: actual - predicted (lakh)")
    plt.title("Residuals versus Predicted Price")
    path = output_dir / "02_residuals.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=40, kde=True)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Residual (lakh)")
    plt.title("Prediction Error Distribution")
    path = output_dir / "03_error_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/used_car_features.csv"),
        help="Path to the engineered feature CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_baseline.joblib"),
        help="Path to the selected model bundle.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/model_evaluation.json"),
        help="Path for evaluation results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation_figures"),
        help="Directory for evaluation plots.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    x_train_transformed = make_dense(preprocessor.transform(x_train))
    x_test_transformed = make_dense(preprocessor.transform(x_test))
    train_predictions = model.predict(x_train_transformed)
    test_predictions = model.predict(x_test_transformed)
    residuals = y_test.to_numpy() - test_predictions

    feature_names = preprocessor.get_feature_names_out()
    importance_sample_size = min(5000, len(x_test_transformed))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(
        len(x_test_transformed), size=importance_sample_size, replace=False
    )
    importance = permutation_importance(
        model,
        x_test_transformed[sample_indices],
        y_test.iloc[sample_indices],
        scoring="neg_root_mean_squared_error",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_path = args.report.with_name("permutation_importance.csv")
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(importance_path, index=False)

    plot_paths = save_plots(y_test, test_predictions, residuals, args.output_dir)
    report = {
        "model": bundle.get("model_name", "selected_model"),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_metrics": calculate_metrics(y_train, train_predictions),
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "residual_quantiles_lakh": {
            str(q): float(np.quantile(residuals, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]
        },
        "permutation_importance_sample_rows": importance_sample_size,
        "top_features": importance_frame.head(15)["feature"].tolist(),
        "importance_file": str(importance_path),
        "plots": plot_paths,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Model: {report['model']}")
    print(f"Test MAE: {report['test_metrics']['mae']:.4f} lakh")
    print(f"Test RMSE: {report['test_metrics']['rmse']:.4f} lakh")
    print(f"Test R2: {report['test_metrics']['r2']:.4f}")
    print(f"Evaluation report written to {args.report}")
    print(f"Permutation importance written to {importance_path}")


if __name__ == "__main__":
    main()
