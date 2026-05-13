from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

# --------------------------------------------------
# App Setup
# --------------------------------------------------
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# Load Model + Scaler
# --------------------------------------------------
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# --------------------------------------------------
# Home Page (UI)
# --------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Ames House Price Predictor</title>
<style>
body {
    font-family: Arial;
    background: #0f172a;
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
.container {
    background: #1e293b;
    padding: 30px;
    border-radius: 15px;
    width: 500px;
}
input {
    width: 100%;
    margin: 5px 0;
    padding: 10px;
}
button {
    width: 100%;
    padding: 10px;
    background: #38bdf8;
    border: none;
    color: black;
    font-weight: bold;
    cursor: pointer;
}
.result {
    margin-top: 20px;
    padding: 15px;
    background: #334155;
    border-radius: 10px;
}
</style>
</head>

<body>
<div class="container">
<h2>🏡 Ames House Price Predictor (FastAPI + XGBoost)</h2>

<form action="/predict" method="post">

<input name="OverallQual" placeholder="Overall Quality" required>
<input name="GrLivArea" placeholder="Above Ground Living Area" required>
<input name="GarageCars" placeholder="Garage Cars" required>
<input name="TotalBsmtSF" placeholder="Total Basement SF" required>
<input name="YearBuilt" placeholder="Year Built" required>
<input name="FullBath" placeholder="Full Bathrooms" required>
<input name="Fireplaces" placeholder="Fireplaces" required>
<input name="LotArea" placeholder="Lot Area" required>

<button type="submit">Predict Price</button>
</form>

{% if prediction %}
<div class="result">
<h3>Log Price: {{ prediction }}</h3>
<h4>Actual Price: {{ actual_price }}</h4>
</div>
{% endif %}

</div>
</body>
</html>
"""

# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# --------------------------------------------------
# Prediction Route (Form)
# --------------------------------------------------
@app.post("/predict")
async def predict(request: Request):

    form = await request.form()

    features = np.array([[ 
        float(form["OverallQual"]),
        float(form["GrLivArea"]),
        float(form["GarageCars"]),
        float(form["TotalBsmtSF"]),
        float(form["YearBuilt"]),
        float(form["FullBath"]),
        float(form["Fireplaces"]),
        float(form["LotArea"])
    ]])

    scaled_features = scaler.transform(features)
    pred_log = model.predict(scaled_features)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(pred_log, 3),
            "actual_price": round(np.expm1(pred_log), 2)
        }
    )

# --------------------------------------------------
# API Endpoint (JSON)
# --------------------------------------------------
@app.post("/predict_api")
async def predict_api(data: dict):

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

    scaled_features = scaler.transform(features)
    pred_log = model.predict(scaled_features)[0]

    return JSONResponse({
        "log_price": round(float(pred_log), 3),
        "predicted_price": round(float(np.expm1(pred_log)), 2),
        "status": "success"
    })
