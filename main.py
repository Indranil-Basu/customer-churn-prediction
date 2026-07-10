from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import Literal


app = FastAPI()

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
from pydantic import BaseModel
from typing import Literal

class Customer(BaseModel):
    gender: Literal['Male', 'Female']
    SeniorCitizen: int
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    MonthlyCharges: float
    TotalCharges: float
    
    
   
    
@app.post("/predict")
def predict_churn(customer: Customer):
    gender = 1 if customer.gender == "Female" else 0
    Partner = 1 if customer.Partner == "Yes" else 0   
    dependents = 1 if customer.Dependents == 'Yes' else 0
    phone_service = 1 if customer.PhoneService == 'Yes' else 0
    paperless_billing = 1 if customer.PaperlessBilling == 'Yes' else 0
    
    contract_map = {'Month-to-month':0, 'One year':1,'Two year':2}
    contract = contract_map[customer.Contract] 
    
    #OnlineSecurity    
    
    online_security_yes = 1 if customer.OnlineSecurity == 'Yes' else 0
    online_security_no_internet = 1 if customer.OnlineSecurity == 'No internet service' else 0
    
    # OnlineBackup
    online_backup_yes = 1 if customer.OnlineBackup == 'Yes' else 0
    online_backup_no_internet = 1 if customer.OnlineBackup == 'No internet service' else 0
    
    # DeviceProtection
    device_protection_yes = 1 if customer.DeviceProtection == 'Yes' else 0
    device_protection_no_internet = 1 if customer.DeviceProtection == 'No internet service' else 0
    
    # TechSupport
    tech_support_yes = 1 if customer.TechSupport == 'Yes' else 0
    tech_support_no_internet = 1 if customer.TechSupport == 'No internet service' else 0
    
    # StreamingTV
    streaming_tv_yes = 1 if customer.StreamingTV == 'Yes' else 0
    streaming_tv_no_internet = 1 if customer.StreamingTV == 'No internet service' else 0
    
    # StreamingMovies
    streaming_movies_yes = 1 if customer.StreamingMovies == 'Yes' else 0
    streaming_movies_no_internet = 1 if customer.StreamingMovies == 'No internet service' else 0
    
    # MultipleLines
    multiple_lines_yes = 1 if customer.MultipleLines == 'Yes' else 0
    multiple_lines_no_phone = 1 if customer.MultipleLines == 'No phone service' else 0
    
    # InternetService
    internet_fiber = 1 if customer.InternetService == 'Fiber optic' else 0
    internet_no = 1 if customer.InternetService == 'No' else 0
    
    # PaymentMethod
    payment_credit_card = 1 if customer.PaymentMethod == 'Credit card (automatic)' else 0
    payment_electronic_check = 1 if customer.PaymentMethod == 'Electronic check' else 0
    payment_mailed_check = 1 if customer.PaymentMethod == 'Mailed check' else 0
    
    
    numeric_feature = np.array([[customer.tenure , customer.MonthlyCharges , customer.TotalCharges]])
    
    scaled_numeric = scaler.transform(numeric_feature)
    
    tenure_scaled = scaled_numeric[0][0]
    monthly_charges_scaled = scaled_numeric[0][1]
    total_charges_scaled = scaled_numeric[0][2]
    
    
    input_data = np.array([[
    gender,
    customer.SeniorCitizen,
    Partner,
    dependents,
    tenure_scaled,
    phone_service,
    contract,
    paperless_billing,
    monthly_charges_scaled,
    total_charges_scaled,
    online_security_no_internet,
    online_security_yes,
    online_backup_no_internet,
    online_backup_yes,
    device_protection_no_internet,
    device_protection_yes,
    tech_support_no_internet,
    tech_support_yes,
    streaming_tv_no_internet,
    streaming_tv_yes,
    streaming_movies_no_internet,
    streaming_movies_yes,
    multiple_lines_no_phone,
    multiple_lines_yes,
    internet_fiber,
    internet_no,
    payment_credit_card,
    payment_electronic_check,
    payment_mailed_check
]])
    
    churn_probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if churn_probability >= 0.35 else 0
    
    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(churn_probability),4),
        "message": "Customer likely to churn" if prediction == 1 else "Customer likely to stay"
    }
    
    
    