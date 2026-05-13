from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.post("/predict_api")
def predict(data: dict):

    features = np.array([[ 
        float(data["OverallQual"]),
        float(data["GrLivArea"]),
        float(data["GarageCars"]),
        float(data["TotalBsmtSF"]),
        float(data["YearBuilt"]),
        float(data["FullBath"]),
        float(data["Fireplaces"]),
        float(data["LotArea"])
    ]])

    scaled = scaler.transform(features)

    pred_log = model.predict(scaled)[0]

    return {
        "log_price": round(float(pred_log),3),
        "price": round(float(np.expm1(pred_log)),2)
    }
