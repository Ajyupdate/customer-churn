from fastapi import FastAPI, Body
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load model and feature columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")


class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    InternetService_Fiber_optic: int
    OnlineSecurity_Yes: int
    OnlineBackup_Yes: int
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaperlessBilling_Yes: int = 0
    PaymentMethod_Electronic_check: int = 0


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running 🚀"}


@app.post("/predict")
def predict(
    data: CustomerData = Body(
        ...,
        examples={
            "non_churn_example": {
                "summary": "Example of a likely NON-CHURN customer",
                "description": "This example represents a loyal customer with long tenure and moderate charges.",
                "value": {
                    "tenure": 60,
                    "MonthlyCharges": 50.0,
                    "TotalCharges": 3000.0,
                    "gender_Male": 1,
                    "Partner_Yes": 1,
                    "Dependents_Yes": 1,
                    "PhoneService_Yes": 1,
                    "InternetService_Fiber_optic": 0,
                    "OnlineSecurity_Yes": 1,
                    "OnlineBackup_Yes": 1,
                    "Contract_One_year": 1,
                    "Contract_Two_year": 0,
                    "PaperlessBilling_Yes": 0,
                    "PaymentMethod_Electronic_check": 0
                },
            },
            "churn_example": {
                "summary": "Example of a LIKELY CHURN customer",
                "description": "This example represents a new customer with high charges and no contract.",
                "value": {
                    "tenure": 2,
                    "MonthlyCharges": 95.5,
                    "TotalCharges": 191.0,
                    "gender_Male": 1,
                    "Partner_Yes": 0,
                    "Dependents_Yes": 0,
                    "PhoneService_Yes": 1,
                    "InternetService_Fiber_optic": 1,
                    "OnlineSecurity_Yes": 0,
                    "OnlineBackup_Yes": 0,
                    "Contract_One_year": 0,
                    "Contract_Two_year": 0,
                    "PaperlessBilling_Yes": 1,
                    "PaymentMethod_Electronic_check": 1
                },
            },
        },
    )
):
    df = pd.DataFrame([data.dict()])

    # Ensure all model columns exist
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    prediction = model.predict(df)[0]

    return {"churn_prediction": int(prediction)}
