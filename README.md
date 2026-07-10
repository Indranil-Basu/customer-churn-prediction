# Customer Churn Prediction API

A machine learning API that predicts telecom customer churn, built with Scikit-learn and served via FastAPI. Unlike a standard classification task, this project required handling class imbalance and making a deliberate precision/recall trade-off based on business context.

## What It Does
Takes customer account details (contract type, tenure, services used, billing info, etc.) and predicts the likelihood of churn, along with a probability score.

## Tech Stack
- Python, Pandas, Scikit-learn
- FastAPI + Uvicorn
- Joblib (model + scaler persistence)
- Postman (API testing)

## Key Technical Decisions

**Hidden data quality issue:** `TotalCharges` appeared to have zero missing values via `.info()`, but contained 11 blank-string values invisible to standard null checks — caught via explicit string inspection, then converted properly using `pd.to_numeric(errors='coerce')`.

**Mixed categorical encoding:** used binary mapping for true Yes/No columns, ordinal mapping for `Contract` (preserving its natural order: Month-to-month < One year < Two year), and one-hot encoding for multi-category columns with no inherent order (`InternetService`, `PaymentMethod`), including columns with a "No internet/phone service" third category.

**Feature scaling:** applied `StandardScaler` to `tenure`, `MonthlyCharges`, and `TotalCharges` after the model failed to converge — these columns had far larger numeric ranges than the mostly-binary features, and scaling resolved the convergence warning.

**Class imbalance handling:** the dataset is ~73.5% "stayed" vs 26.5% "churned." Used `stratify=y` in the train/test split to preserve this ratio in both sets, and relied on precision/recall/F1 (not just accuracy) to evaluate the model fairly.

**Custom decision threshold:** given that failing to identify a churning customer is more costly to the business than a false alarm, I lowered the classification threshold from the default 0.5 to 0.35. This improved recall for the churn class from 57% to 72%, at the cost of precision dropping from 65% to 54% and overall accuracy dropping from 81% to 77% — a deliberate trade-off aligned with the business priority of catching more at-risk customers.

## Model Performance (at 0.35 threshold)
| Metric | Class: Stayed | Class: Churned |
|---|---|---|
| Precision | 0.88 | 0.54 |
| Recall | 0.78 | 0.72 |
| F1-score | 0.83 | 0.62 |

Overall accuracy: 77% (vs. 73.5% baseline of always predicting "stayed")

## Project Structure

    customer-churn-prediction/
    ├── data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    ├── 01_explore_data.ipynb
    ├── main.py
    ├── churn_model.pkl
    ├── scaler.pkl
    └── requirements.txt

## How to Run
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`
6. Visit `http://127.0.0.1:8000/docs` to test interactively

## Example Request
POST `/predict`

    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 90.0,
      "TotalCharges": 90.0
    }

## Response

    {
      "churn_prediction": 1,
      "churn_probability": 0.6331,
      "message": "Customer likely to churn"
    }

## Future Improvements
- Try ensemble models (Random Forest, XGBoost) to improve precision without sacrificing recall
- Explore resampling techniques (SMOTE) for handling class imbalance
