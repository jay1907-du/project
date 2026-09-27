from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

app = FastAPI(title="DAM EWS API")


# =========================
# Input data structure
# =========================
class PredictionInput(BaseModel):
    water_level: float
    water_level_change: float
    rainfall_1d: float
    rainfall_3d: float
    rainfall_7d: float


# =========================
# Load DAM EWS data
# =========================
reservoir_file = "reservior_water_level_manual_daily_cwc_mp_1970_2025.csv"
rainfall_file = "rainfall_manual_daily_madhya_pradesh_sw_mp_2021_2025.csv"

reservoir = pd.read_csv(reservoir_file)
rainfall = pd.read_csv(rainfall_file)


# =========================
# Prepare reservoir data
# =========================
reservoir["Data Acquisition Time"] = pd.to_datetime(
    reservoir["Data Acquisition Time"],
    errors="coerce"
)

reservoir["date"] = reservoir["Data Acquisition Time"].dt.date

daily_reservoir = (
    reservoir.groupby("date")
    .agg({
        "Manual Daily Reservoir water level (m)": "mean"
    })
    .reset_index()
)

daily_reservoir["date"] = pd.to_datetime(
    daily_reservoir["date"]
)


# =========================
# Create water level change
# =========================
daily_reservoir["water_level_change"] = (
    daily_reservoir[
        "Manual Daily Reservoir water level (m)"
    ].diff()
)


# =========================
# Prepare rainfall data
# =========================
rainfall["Data Acquisition Time"] = pd.to_datetime(
    rainfall["Data Acquisition Time"],
    errors="coerce"
)

rainfall["date"] = rainfall["Data Acquisition Time"].dt.date

rainfall_daily = (
    rainfall.groupby("date")
    .agg({
        "Manual Daily Rainfall (mm)": "mean"
    })
    .reset_index()
)

rainfall_daily = rainfall_daily.rename(
    columns={
        "Manual Daily Rainfall (mm)": "rainfall_1d"
    }
)

rainfall_daily["date"] = pd.to_datetime(
    rainfall_daily["date"]
)


# =========================
# Rainfall rolling features
# =========================
rainfall_daily["rainfall_3d"] = (
    rainfall_daily["rainfall_1d"]
    .rolling(3)
    .sum()
)

rainfall_daily["rainfall_7d"] = (
    rainfall_daily["rainfall_1d"]
    .rolling(7)
    .sum()
)


# =========================
# Merge data
# =========================
final_df = pd.merge(
    daily_reservoir,
    rainfall_daily,
    on="date",
    how="inner"
)

final_df = final_df.dropna()


# =========================
# ML features
# =========================
features = [
    "Manual Daily Reservoir water level (m)",
    "water_level_change",
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d"
]

X = final_df[features]

y = final_df[
    "Manual Daily Reservoir water level (m)"
].shift(-1)

valid = y.notna()

X = X[valid]
y = y[valid]


# =========================
# Train model
# =========================
model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

print("DAM EWS model trained successfully")


# =========================
# Prediction API
# =========================
@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{
        "Manual Daily Reservoir water level (m)": data.water_level,
        "water_level_change": data.water_level_change,
        "rainfall_1d": data.rainfall_1d,
        "rainfall_3d": data.rainfall_3d,
        "rainfall_7d": data.rainfall_7d
    }])

    next_day_prediction = model.predict(
        input_data
    )[0]

    predicted_change = (
        next_day_prediction - data.water_level
    )

    predicted_level_percent = (
        (next_day_prediction - 403.55)
        / (422.76 - 403.55)
    ) * 100

    predicted_risk_score = (
        0.6 * predicted_level_percent
        +
        0.4 * max(0, predicted_change)
        / 0.15 * 100
    )

    predicted_risk_score = max(
        0,
        min(100, predicted_risk_score)
    )

    if predicted_risk_score < 40:
        predicted_risk = "GREEN"

    elif predicted_risk_score < 70:
        predicted_risk = "YELLOW"

    else:
        predicted_risk = "RED"

    return {
        "next_day_prediction": round(
            float(next_day_prediction), 3
        ),
        "predicted_change": round(
            float(predicted_change), 3
        ),
        "predicted_risk_score": round(
            float(predicted_risk_score), 2
        ),
        "predicted_risk": predicted_risk
    }


# =========================
# Home endpoint
# =========================
@app.get("/")
def home():
    return {
        "message": "DAM EWS API is running"
    }
