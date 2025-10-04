from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


app = FastAPI(title="Exoplanet Classification API", description="Predict exoplanet disposition using LightGBM model")

# Add CORS for Flutter
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter app’s origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model and feature names
model = joblib.load('exoplanet_model.pkl')
feature_names = joblib.load('feature_names.pkl')
imputer = joblib.load('imputer.pkl')

# Define the input model matching your features
class PredictionInput(BaseModel):
    koi_fpflag_nt: Optional[int] = 0
    koi_fpflag_ss: Optional[int] = 0
    koi_fpflag_co: Optional[int] = 0
    koi_fpflag_ec: Optional[int] = 0
    koi_depth: float
    koi_duration: float
    koi_impact: float
    koi_period: float
    koi_model_snr: float
    koi_prad: float
    koi_teq: float
    koi_steff: float
    koi_srad: float
    koi_slogg: float


# Define a model for each probability entry
class Probability(BaseModel):
    class_id: int
    probability: float
    
    
# Define the output model
class PredictionOutput(BaseModel):
    prediction: int
    class_label: str
    probabilities: List[Probability]  # Updated to match the response structure
    
@app.post("/predict", response_model=PredictionOutput)
async def predict_disposition(input_data: PredictionInput):
    try:
        # Prepare the input as a DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Ensure columns match feature_names (in the order from training)
        input_df = input_df.reindex(columns=feature_names)
        
        # Apply SimpleImputer to numerical features
        numerical_features = [
            'koi_depth', 'koi_impact', 'koi_model_snr', 'koi_prad', 'koi_teq',
            'koi_steff', 'koi_srad', 'koi_slogg'
        ]
        input_df[numerical_features] = pd.DataFrame(
            imputer.transform(input_df[numerical_features]),
            columns=numerical_features,
            index=input_df.index
        )
        # Convert to numpy array
        X_input = input_df.values.astype(np.float32)
        
        # Predict
        pred = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0].tolist()
        
        # Map prediction back to label
        labels = {0: 'CONFIRMED (Exoplanet)', 1: 'CANDIDATE', 2: 'FALSE POSITIVE'}
        class_label = labels[pred]
        
        
        # Return probabilities as a list of dicts
        probabilities = [{"class_id": i, "probability": p} for i, p in enumerate(probs)]
        
        return PredictionOutput(
            prediction=pred,
            class_label=class_label,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Exoplanet Classification API is running!"}

# To run: uvicorn main:app --reload
# Visit http://127.0.0.1:8000/docs for interactive API docs