# BankWise — Bank Customer Analytics

A beginner-friendly full-stack machine learning project using the supplied Churn_Modelling.csv dataset.

## Three models
1. Linear Regression: estimates account balance; report R² and MAE.
2. Logistic Regression: predicts customer churn; report accuracy.
3. Decision Tree Classifier: predicts churn using a second, more interpretable model; report accuracy.

## Run on Windows / VS Code

1. Extract this folder.
2. Open the folder in VS Code.
3. Open Terminal → New Terminal.
4. (Optional but recommended) create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the app:
   ```bash
   uvicorn main:app --reload
   ```
7. Open http://127.0.0.1:8000
8. API documentation: http://127.0.0.1:8000/docs

## Retrain models
The trained model files are included. To retrain, run:
```bash
python train_models.py
```

## API endpoints
- POST `/predict/regression`
- POST `/predict/classification`
- POST `/predict/tree`
- GET `/api/metrics`

Each prediction endpoint expects customer fields:
`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `NumOfProducts`,
`HasCrCard`, `IsActiveMember`, `EstimatedSalary`.

## Important notes
- The dataset has 10,000 records and `Exited` indicates whether the customer left.
- `Balance` is used as the numeric target for the regression demonstration. This is a modelling exercise, not a forecast of bank profit.
- Churn predictions are estimates and should not be treated as certain outcomes.
- The metrics are calculated on a held-out 20% test set.
- Understand the code before your viva; the assignment asks you to explain your implementation.
