"""Production prediction helpers for the saved used-car model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026
TARGET_COLUMN = "Selling_Price_Lakh"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_baseline.joblib"
DEFAULT_EVALUATION_PATH = Path(__file__).resolve().parents[1] / "reports" / "model_evaluation.json"


def load_model_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(model_path)


def load_prediction_residuals(evaluation_path: Path = DEFAULT_EVALUATION_PATH) -> tuple[float, float]:
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    quantiles = report["residual_quantiles_lakh"]
    return float(quantiles["0.05"]), float(quantiles["0.95"])


def prepare_input(input_data: dict) -> pd.DataFrame:
    row = dict(input_data)
    row["Car_Age"] = REFERENCE_YEAR - int(row["Year"])
    row["KM_Per_Year"] = float(row["KM_Driven"]) / max(row["Car_Age"], 1)
    row["Car_Age_Squared"] = row["Car_Age"] ** 2
    row["Log_KM_Driven"] = np.log1p(max(float(row["KM_Driven"]), 0))
    row["Log_Present_Price_Lakh"] = np.log1p(max(float(row["Present_Price_Lakh"]), 0))
    row["Condition_Age_Interaction"] = float(row["Condition_Score"]) / max(row["Car_Age"], 1)
    row["Recent_Vehicle"] = int(row["Car_Age"] <= 3)
    return pd.DataFrame([row])


def validate_input(input_data: dict) -> list[str]:
    errors: list[str] = []
    year = int(input_data["Year"])
    if year < 1900 or year > REFERENCE_YEAR:
        errors.append(f"Manufacturing year must be between 1900 and {REFERENCE_YEAR}.")
    if float(input_data["KM_Driven"]) < 0:
        errors.append("Kilometres driven cannot be negative.")
    if float(input_data["Present_Price_Lakh"]) <= 0:
        errors.append("Present price must be greater than zero.")
    if float(input_data["Engine_CC"]) <= 0:
        errors.append("Engine capacity must be greater than zero.")
    if float(input_data["Mileage_KMPL"]) <= 0:
        errors.append("Mileage must be greater than zero.")
    if not 0 <= int(input_data["Owner"]) <= 10:
        errors.append("Owner count must be between 0 and 10.")
    if not 0 <= float(input_data["Condition_Score"]) <= 10:
        errors.append("Condition score must be between 0 and 10.")
    return errors


def predict_price(
    input_data: dict,
    bundle: dict | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
) -> dict[str, float]:
    errors = validate_input(input_data)
    if errors:
        raise ValueError(" ".join(errors))
    bundle = bundle or load_model_bundle(model_path)
    prepared = prepare_input(input_data)
    transformed = bundle["preprocessor"].transform(prepared)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    prediction = float(bundle["model"].predict(transformed)[0])
    lower_residual, upper_residual = load_prediction_residuals(evaluation_path)
    return {
        "prediction_lakh": max(prediction, 0.0),
        "lower_lakh": max(prediction + lower_residual, 0.0),
        "upper_lakh": max(prediction + upper_residual, 0.0),
    }
