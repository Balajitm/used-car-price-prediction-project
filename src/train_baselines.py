"""Train and compare baseline regression models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import TARGET_COLUMN
from preprocessing import RANDOM_STATE, TEST_SIZE, build_preprocessor


MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),
}


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    non_zero = actual != 0
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": float(mape),
    }


def make_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/used_car_features.csv"),
        help="Path to the engineered feature CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/baseline_results.csv"),
        help="Path for model comparison results.",
    )
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("models/best_baseline.joblib"),
        help="Path for the best baseline bundle.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    dense_models = {"HistGradientBoosting"}
    results: list[dict] = []
    fitted_models: dict[str, object] = {}

    for name, model in MODELS.items():
        train_matrix = make_dense(x_train_transformed) if name in dense_models else x_train_transformed
        test_matrix = make_dense(x_test_transformed) if name in dense_models else x_test_transformed
        started = time.perf_counter()
        model.fit(train_matrix, y_train)
        training_seconds = time.perf_counter() - started
        train_predictions = model.predict(train_matrix)
        test_predictions = model.predict(test_matrix)
        train_metrics = calculate_metrics(y_train, train_predictions)
        test_metrics = calculate_metrics(y_test, test_predictions)
        results.append(
            {
                "model": name,
                "training_seconds": round(training_seconds, 3),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mape_percent": test_metrics["mape_percent"],
            }
        )
        fitted_models[name] = model
        print(f"{name}: test RMSE={test_metrics['rmse']:.4f}, test R2={test_metrics['r2']:.4f}")

    results_frame = pd.DataFrame(results).sort_values("test_rmse")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.best_model.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(args.results, index=False)

    best_name = str(results_frame.iloc[0]["model"])
    joblib.dump(
        {
            "model": fitted_models[best_name],
            "preprocessor": preprocessor,
            "model_name": best_name,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "target_column": TARGET_COLUMN,
        },
        args.best_model,
    )
    summary = {
        "best_model_by_test_rmse": best_name,
        "results_file": str(args.results),
        "model_file": str(args.best_model),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    summary_path = args.results.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best baseline: {best_name}")
    print(f"Results written to {args.results}")
    print(f"Model bundle written to {args.best_model}")


if __name__ == "__main__":
    main()
