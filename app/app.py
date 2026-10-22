"""Streamlit interface for used-car price estimation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_model_bundle, predict_price

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "used_car_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_baseline.joblib"
EVALUATION_PATH = PROJECT_ROOT / "reports" / "model_evaluation.json"


@st.cache_data
def load_options() -> dict[str, list[str]]:
	data = pd.read_csv(DATA_PATH)
	columns = [
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
	return {column: sorted(data[column].dropna().astype(str).unique().tolist()) for column in columns}


@st.cache_resource
def load_bundle() -> dict:
	return load_model_bundle(MODEL_PATH)


def render_vehicle_inputs(options: dict[str, list[str]], key_prefix: str) -> dict:
	left, right = st.columns(2)
	with left:
		brand = st.selectbox("Brand", options["Brand"], key=f"{key_prefix}_brand")
		model = st.selectbox("Model", options["Model"], key=f"{key_prefix}_model")
		year = st.number_input("Manufacturing year", min_value=2012, max_value=2026, value=2021, step=1, key=f"{key_prefix}_year")
		kilometres = st.number_input("Kilometres driven", min_value=0, max_value=500000, value=45000, step=1000, key=f"{key_prefix}_kilometres")
		owner = st.number_input("Previous owners", min_value=0, max_value=10, value=1, step=1, key=f"{key_prefix}_owner")
		fuel = st.selectbox("Fuel type", options["Fuel_Type"], key=f"{key_prefix}_fuel")
		transmission = st.selectbox("Transmission", options["Transmission"], key=f"{key_prefix}_transmission")
	with right:
		location = st.selectbox("Location", options["Location"], key=f"{key_prefix}_location")
		present_price = st.number_input("Original / present price (lakh)", min_value=0.1, max_value=100.0, value=18.0, step=0.1, key=f"{key_prefix}_present_price")
		engine = st.number_input("Engine capacity (cc)", min_value=100, max_value=6000, value=1600, step=50, key=f"{key_prefix}_engine")
		mileage = st.number_input("Mileage (kmpl)", min_value=1.0, max_value=100.0, value=15.0, step=0.1, key=f"{key_prefix}_mileage")
		condition = st.slider("Condition score", min_value=0.0, max_value=10.0, value=8.0, step=0.1, key=f"{key_prefix}_condition")
		insurance = st.selectbox("Insurance valid", options["Insurance_Valid"], key=f"{key_prefix}_insurance")
		accident = st.selectbox("Accident history", options["Accident_History"], key=f"{key_prefix}_accident")
		service = st.selectbox("Service history", options["Service_History"], key=f"{key_prefix}_service")
		seller = st.selectbox("Seller type", options["Dealer_or_Individual"], key=f"{key_prefix}_seller")
	return {
		"Brand": brand,
		"Model": model,
		"Year": year,
		"KM_Driven": kilometres,
		"Fuel_Type": fuel,
		"Transmission": transmission,
		"Owner": owner,
		"Location": location,
		"Present_Price_Lakh": present_price,
		"Insurance_Valid": insurance,
		"Accident_History": accident,
		"Service_History": service,
		"Condition_Score": condition,
		"Engine_CC": engine,
		"Mileage_KMPL": mileage,
		"Dealer_or_Individual": seller,
	}


def render_result(label: str, result: dict[str, float]) -> None:
	st.subheader(label)
	metric_left, metric_middle, metric_right = st.columns(3)
	metric_left.metric("Estimated price", f"₹{result['prediction_lakh']:.2f} lakh")
	metric_middle.metric("Lower estimate", f"₹{result['lower_lakh']:.2f} lakh")
	metric_right.metric("Upper estimate", f"₹{result['upper_lakh']:.2f} lakh")


def main() -> None:
	st.set_page_config(page_title="Used Car Price Prediction", page_icon="₹", layout="wide")
	st.title("Used Car Price Prediction")
	st.caption("Estimate resale value from vehicle specifications and current condition.")

	if not MODEL_PATH.exists() or not EVALUATION_PATH.exists():
		st.error("The trained model or evaluation report is missing. Run the training workflow first.")
		st.stop()

	options = load_options()
	mode = st.radio("Mode", ["Single estimate", "Compare two cars"], horizontal=True)
	with st.form("prediction_form"):
		if mode == "Compare two cars":
			car_left, car_right = st.columns(2)
			with car_left:
				st.subheader("Car A")
				car_a = render_vehicle_inputs(options, "car_a")
			with car_right:
				st.subheader("Car B")
				car_b = render_vehicle_inputs(options, "car_b")
		else:
			st.subheader("Vehicle details")
			car_a = render_vehicle_inputs(options, "single")
			car_b = None
		submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

	if submitted:
		try:
			results = [
				predict_price(car_a, bundle=load_bundle(), evaluation_path=EVALUATION_PATH),
			]
			if car_b is not None:
				results.append(predict_price(car_b, bundle=load_bundle(), evaluation_path=EVALUATION_PATH))
		except (KeyError, ValueError, FileNotFoundError) as error:
			st.error(str(error))
		else:
			st.success("Estimate generated")
			if car_b is None:
				render_result("Estimated resale value", results[0])
			else:
				result_frame = pd.DataFrame(
					[
						{"Vehicle": "Car A", "Estimated price (lakh)": results[0]["prediction_lakh"], "Lower (lakh)": results[0]["lower_lakh"], "Upper (lakh)": results[0]["upper_lakh"]},
						{"Vehicle": "Car B", "Estimated price (lakh)": results[1]["prediction_lakh"], "Lower (lakh)": results[1]["lower_lakh"], "Upper (lakh)": results[1]["upper_lakh"]},
					]
				)
				st.dataframe(result_frame, hide_index=True, use_container_width=True)
				st.bar_chart(result_frame.set_index("Vehicle")["Estimated price (lakh)"])
			st.caption("The range is empirical and reflects residuals observed on the held-out test set.")
			st.subheader("Main model signals")
			st.write("Present price, vehicle age and condition, manufacturing year, kilometres driven, and engine capacity were the strongest signals in the evaluation.")


if __name__ == "__main__":
	main()
