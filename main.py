from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title="Bank Customer Analytics", version="1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

balance_model = joblib.load(MODEL_DIR / "balance_regression.joblib")
logistic_model = joblib.load(MODEL_DIR / "churn_logistic.joblib")
tree_model = joblib.load(MODEL_DIR / "churn_decision_tree.joblib")
metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

class CustomerInput(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0, le=250000)

def customer_frame(customer):
    data = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    if data["Geography"] not in ["France", "Spain", "Germany"]:
        raise HTTPException(status_code=422, detail="Country must be France, Spain, or Germany.")
    if data["Gender"] not in ["Female", "Male"]:
        raise HTTPException(status_code=422, detail="Gender must be Female or Male.")
    return pd.DataFrame([data])

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/metrics")
def get_metrics():
    return metrics

@app.post("/predict/regression")
def predict_balance(customer: CustomerInput):
    row = customer_frame(customer)
    prediction = float(balance_model.predict(row)[0])
    return {"predicted_balance": round(max(0, prediction), 2)}

@app.post("/predict/classification")
def predict_churn_logistic(customer: CustomerInput):
    row = customer_frame(customer)
    prediction = int(logistic_model.predict(row)[0])
    probability = float(logistic_model.predict_proba(row)[0][1])
    return {
        "model": "Logistic Regression",
        "prediction": "Likely to leave" if prediction == 1 else "Likely to stay",
        "churn_probability": round(probability * 100, 2),
        "business_tip": "Consider a retention follow-up." if prediction == 1 else "Continue regular customer engagement."
    }

@app.post("/predict/tree")
def predict_churn_tree(customer: CustomerInput):
    row = customer_frame(customer)
    prediction = int(tree_model.predict(row)[0])
    probability = float(tree_model.predict_proba(row)[0][1])

    # Read the decision rules the tree followed for this customer.
    preprocessor = tree_model.named_steps["preprocessor"]
    decision_tree = tree_model.named_steps["model"]
    transformed_row = preprocessor.transform(row)
    path = decision_tree.decision_path(transformed_row)
    feature_names = preprocessor.get_feature_names_out()
    rules = []
    for node_id in path.indices:
        feature_index = decision_tree.tree_.feature[node_id]
        if feature_index >= 0:
            feature_name = str(feature_names[feature_index]).replace("numeric__", "").replace("categorical__", "")
            threshold = float(decision_tree.tree_.threshold[node_id])
            rules.append(f"{feature_name} was {'at or below' if transformed_row[0, feature_index] <= threshold else 'above'} {threshold:.2f}")
    return {
        "model": "Decision Tree",
        "prediction": "Likely to leave" if prediction == 1 else "Likely to stay",
        "churn_probability": round(probability * 100, 2),
        "decision_rules": rules[:4],
        "business_tip": "Review this customer's needs and consider a retention offer." if prediction == 1 else "Maintain the current customer relationship."
    }
