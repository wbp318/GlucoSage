from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, confloat
from typing import List
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()

class GlucoseData(BaseModel):
    current_glucose: confloat(gt=0)
    target_glucose: confloat(gt=0)
    insulin_sensitivity: confloat(gt=0)
    carb_ratio: confloat(gt=0)
    carbs_to_eat: confloat(ge=0)
    time: datetime

class InsulinDose(BaseModel):
    correction_dose: float
    meal_dose: float
    total_dose: float
    predicted_glucose: float

class GlucoseHistory(BaseModel):
    glucose_levels: List[float]
    timestamps: List[datetime]

glucose_predictor = joblib.load('glucose_predictor_model.joblib')

def calculate_doses(data: GlucoseData):
    correction_dose = (data.current_glucose - data.target_glucose) / data.insulin_sensitivity
    meal_dose = data.carbs_to_eat / data.carb_ratio
    total_dose = max(0, correction_dose) + meal_dose
    return correction_dose, meal_dose, total_dose

def predict_future_glucose(data: GlucoseData, total_dose: float):
    # Prepare input data for the predictor model
    input_data = pd.DataFrame({
        'hour': [data.time.hour],
        'day_of_week': [data.time.weekday()],
        'insulin_dose': [total_dose],
        'carb_intake': [data.carbs_to_eat],
        'glucose_lag1': [data.current_glucose],
        'glucose_lag2': [data.current_glucose],  # Simplified, should use actual history
        'glucose_diff': [0],  # Simplified
        'insulin_lag1': [0]  # Simplified
    })
    
    predicted_glucose = glucose_predictor.predict(input_data)[0]
    return predicted_glucose

@app.post("/calculate_insulin_dose", response_model=InsulinDose)
async def calculate_insulin_dose(data: GlucoseData):
    correction_dose, meal_dose, total_dose = calculate_doses(data)
    predicted_glucose = predict_future_glucose(data, total_dose)

    return InsulinDose(
        correction_dose=round(correction_dose, 2),
        meal_dose=round(meal_dose, 2),
        total_dose=round(total_dose, 2),
        predicted_glucose=round(predicted_glucose, 2)
    )

@app.post("/predict_glucose")
async def predict_glucose(history: GlucoseHistory):
    if len(history.glucose_levels) < 2:
        raise HTTPException(status_code=400, detail="Insufficient glucose history")
    
    # Prepare input data for the predictor model
    input_data = pd.DataFrame({
        'hour': [history.timestamps[-1].hour],
        'day_of_week': [history.timestamps[-1].weekday()],
        'insulin_dose': [0],  # Assuming no insulin dose for prediction
        'carb_intake': [0],   # Assuming no carb intake for prediction
        'glucose_lag1': [history.glucose_levels[-1]],
        'glucose_lag2': [history.glucose_levels[-2]],
        'glucose_diff': [history.glucose_levels[-1] - history.glucose_levels[-2]],
        'insulin_lag1': [0]  # Simplified
    })