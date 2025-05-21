import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model and encoders
model = joblib.load("model.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request data model
class InputData(BaseModel):
    Statement: str
    Web: str
    Category: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Machine Learning API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)

    # Convert encoded label back to original
    predicted_label = target_encoder.inverse_transform(prediction)[0]

    return {"prediction": predicted_label}




