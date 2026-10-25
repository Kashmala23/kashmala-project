"""Train and save the three simple models used by BankWise."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

BASE = Path(__file__).resolve().parent
DATA = pd.read_csv(BASE / "Churn_Modelling.csv")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
NUMERIC = ["CreditScore", "Age", "Tenure", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]

def make_preprocessor(scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL)
    ])

X = DATA[FEATURES]
X_train, X_test, y_train, y_test = train_test_split(
    X, DATA["Balance"], test_size=0.2, random_state=42
)
regression = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", LinearRegression())
])
regression.fit(X_train, y_train)
balance_prediction = regression.predict(X_test)
metrics = {"regression": {
    "r2_score": round(float(r2_score(y_test, balance_prediction)), 4),
    "mae": round(float(mean_absolute_error(y_test, balance_prediction)), 2)
}}
joblib.dump(regression, MODEL_DIR / "balance_regression.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, DATA["Exited"], test_size=0.2, random_state=42, stratify=DATA["Exited"]
)
logistic = Pipeline([
    ("preprocessor", make_preprocessor(scale=True)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
logistic.fit(X_train, y_train)
metrics["logistic_regression"] = {
    "accuracy": round(float(accuracy_score(y_test, logistic.predict(X_test))), 4)
}
joblib.dump(logistic, MODEL_DIR / "churn_logistic.joblib")

tree = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))
])
tree.fit(X_train, y_train)
metrics["decision_tree"] = {
    "accuracy": round(float(accuracy_score(y_test, tree.predict(X_test))), 4)
}
joblib.dump(tree, MODEL_DIR / "churn_decision_tree.joblib")
metrics["dataset_rows"] = int(len(DATA))
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
