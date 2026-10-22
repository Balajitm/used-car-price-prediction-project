"""Tune the strongest baseline model without using the test set for selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from feature_engineering import TARGET_COLUMN
from preprocessing import RANDOM_STATE, TEST_SIZE, build_preprocessor


def make_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


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
        default=Path("models/tuned_hist_gradient_boosting.joblib"),
        help="Path for the tuned model bundle.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/tuning_results.json"),
        help="Path for tuning results.",
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
    x_train_transformed = make_dense(preprocessor.fit_transform(x_train))
    x_test_transformed = make_dense(preprocessor.transform(x_test))

    estimator = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    parameter_distributions = {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=2,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=True,
    )

    started = time.perf_counter()
    search.fit(x_train_transformed, y_train)
    search_seconds = time.perf_counter() - started
    best_model = search.best_estimator_
    train_predictions = best_model.predict(x_train_transformed)
    test_predictions = best_model.predict(x_test_transformed)
    train_metrics = metrics(y_train, train_predictions)
    test_metrics = metrics(y_test, test_predictions)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "preprocessor": preprocessor,
            "model_name": "TunedHistGradientBoosting",
            "target_column": TARGET_COLUMN,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        args.model_output,
    )
    report = {
        "model": "TunedHistGradientBoosting",
        "search_type": "RandomizedSearchCV",
        "iterations": 2,
        "cross_validation_folds": 3,
        "search_seconds": round(search_seconds, 3),
        "best_parameters": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_used_for_selection": False,
        "model_output": str(args.model_output),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Tuned model written to {args.model_output}")
    print(f"Tuning report written to {args.report}")


if __name__ == "__main__":
    main()
