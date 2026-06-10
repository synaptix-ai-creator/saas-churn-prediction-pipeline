import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List
from src.data_processor import DataProcessor

# Create FastAPI app
app = FastAPI(
    title="ChurnIQ REST API",
    description="Enterprise SaaS Churn Prediction Inference Service",
    version="1.0.0"
)

# Load production assets
MODEL_PATH = 'models/lightgbm_churn_model.pkl'
FEATURES_PATH = 'models/model_features.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Production model assets not found. Run model training first.")

model = joblib.load(MODEL_PATH)
processor = DataProcessor(FEATURES_PATH)

# Define schemas
class CustomerTelemetry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    customer_id: str = Field(..., alias="Customer ID", description="Unique client identifier", json_schema_extra={"example": "CUS-8924"})
    total_transactions: int = Field(..., description="LTV billing transactions count", json_schema_extra={"example": 24})
    total_revenue: float = Field(..., description="Lifetime Value (LTV) revenue in USD", json_schema_extra={"example": 1250.0})
    currently_auto_renews: int = Field(..., description="Auto-renew status (1: Active, 0: Disabled)", json_schema_extra={"example": 1})
    has_cancelled_before: int = Field(..., description="Cancellation history status (1: Yes, 0: No)", json_schema_extra={"example": 0})
    total_active_days: int = Field(..., description="Active sessions / days in last 30 days", json_schema_extra={"example": 8})
    total_songs_skipped: int = Field(..., description="Feature drop-offs / skipped actions count", json_schema_extra={"example": 60})
    total_songs_completed: int = Field(..., description="Key feature adoptions / completed actions count", json_schema_extra={"example": 45})
    total_listen_time_secs: float = Field(..., description="Product usage duration in seconds", json_schema_extra={"example": 15000.0})

    @model_validator(mode='before')
    @classmethod
    def map_saas_keys(cls, data):
        if not isinstance(data, dict):
            return data
            
        mapping = {
            'customer id': 'Customer ID',
            'customer_id': 'Customer ID',
            'account id': 'Customer ID',
            'account_id': 'Customer ID',
            
            'total_transactions': 'total_transactions',
            'billing cycles': 'total_transactions',
            'billing_cycles': 'total_transactions',
            'transactions': 'total_transactions',
            
            'total_revenue': 'total_revenue',
            'ltv ($)': 'total_revenue',
            'ltv': 'total_revenue',
            'lifetime value': 'total_revenue',
            'lifetime_value': 'total_revenue',
            
            'currently_auto_renews': 'currently_auto_renews',
            'auto-renew status': 'currently_auto_renews',
            'auto_renew_status': 'currently_auto_renews',
            'auto-renew': 'currently_auto_renews',
            'auto_renew': 'currently_auto_renews',
            'auto renew status': 'currently_auto_renews',
            'auto renew': 'currently_auto_renews',
            
            'has_cancelled_before': 'has_cancelled_before',
            'prior cancellations': 'has_cancelled_before',
            'prior_cancellations': 'has_cancelled_before',
            'cancelled before': 'has_cancelled_before',
            'cancelled_before': 'has_cancelled_before',
            
            'total_active_days': 'total_active_days',
            'active sessions (30-day)': 'total_active_days',
            'active_sessions_(30-day)': 'total_active_days',
            'active sessions': 'total_active_days',
            'active_sessions': 'total_active_days',
            
            'total_songs_skipped': 'total_songs_skipped',
            'failed sessions / errors': 'total_songs_skipped',
            'failed_sessions_/_errors': 'total_songs_skipped',
            'failed sessions': 'total_songs_skipped',
            'failed_sessions': 'total_songs_skipped',
            'feature drops': 'total_songs_skipped',
            'feature_drops': 'total_songs_skipped',
            'features skipped/drops': 'total_songs_skipped',
            
            'total_songs_completed': 'total_songs_completed',
            'core feature adoptions': 'total_songs_completed',
            'core_feature_adoptions': 'total_songs_completed',
            'core features used': 'total_songs_completed',
            'core_features_used': 'total_songs_completed',
            'features completed': 'total_songs_completed',
            'features_completed': 'total_songs_completed',
            
            'total_listen_time_secs': 'total_listen_time_secs',
            'product usage duration (mins)': 'total_listen_time_secs',
            'product_usage_duration_(mins)': 'total_listen_time_secs',
            'product usage duration': 'total_listen_time_secs',
            'product_usage_duration': 'total_listen_time_secs',
            'usage duration (mins)': 'total_listen_time_secs',
            'usage_duration_(mins)': 'total_listen_time_secs',
            'usage duration': 'total_listen_time_secs',
        }
        
        standardized_data = {}
        for k, v in data.items():
            cleaned_key = str(k).strip().lower()
            if cleaned_key in mapping:
                target_key = mapping[cleaned_key]
                if target_key == 'total_listen_time_secs':
                    if 'mins' in cleaned_key or 'minute' in cleaned_key or 'duration' in cleaned_key:
                        if 'secs' not in cleaned_key and 'second' not in cleaned_key:
                            v = float(v) * 60.0
                standardized_data[target_key] = v
            else:
                standardized_data[k] = v
                
        return standardized_data

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_churner: bool
    risk_category: str

@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "model": "LightGBM Production"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(telemetry: CustomerTelemetry):
    """
    Calculate churn risk probability and risk status for a single customer.
    """
    try:
        # Convert schema model to dictionary, mapping aliases back to model expectations
        data_dict = telemetry.model_dump(by_alias=False)
        
        # DataProcessor expects pandas DataFrame
        raw_df = pd.DataFrame([data_dict])
        processed_df = processor.prepare_inference_data(raw_df)
        
        # Calculate prediction
        prob = float(model.predict_proba(processed_df)[0][1] * 100)
        
        # Calibrate prediction using business rules
        prob = processor.calibrate_probability(data_dict, prob)
        is_churn = prob >= 50.0
        
        # Map risk categories
        if prob >= 70:
            category = "High Flight Risk"
        elif prob >= 40:
            category = "At Risk"
        else:
            category = "Healthy"
            
        return PredictionResponse(
            customer_id=telemetry.customer_id,
            churn_probability=round(prob, 2),
            is_churner=is_churn,
            risk_category=category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", tags=["Inference"])
def batch_predict_churn(file: UploadFile = File(...)):
    """
    Accepts a CRM telemetry CSV or Excel file, runs batch inference, and returns list of risk assessments.
    """
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")
        
    try:
        if file.filename.endswith('.csv'):
            raw_df = pd.read_csv(file.file)
        else:
            raw_df = pd.read_excel(file.file)
            
        # Standardize columns using DataProcessor
        batch_df = processor.standardize_columns(raw_df)
            
        # Validate required columns
        missing_cols = [col for col in processor.required_features + ['Customer ID'] if col not in batch_df.columns]
        if missing_cols:
            raise ValueError(f"Uploaded schema is missing fields: {', '.join(missing_cols)}")
            
        # Process and Predict
        predict_df = processor.prepare_inference_data(batch_df)
        probabilities = model.predict_proba(predict_df)[:, 1] * 100
        
        results = []
        for idx, row in batch_df.iterrows():
            prob = float(probabilities[idx])
            
            # Calibrate prediction using business rules
            row_dict = row.to_dict()
            prob = processor.calibrate_probability(row_dict, prob)
            is_churn = prob >= 50.0
            
            if prob >= 70:
                cat = "High Flight Risk"
            elif prob >= 40:
                cat = "At Risk"
            else:
                cat = "Healthy"
                
            results.append({
                "Customer ID": row["Customer ID"],
                "churn_probability": round(prob, 2),
                "is_churner": is_churn,
                "risk_category": cat,
                "total_revenue": float(row["total_revenue"])
            })
            
        return {"total_processed": len(results), "accounts": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
