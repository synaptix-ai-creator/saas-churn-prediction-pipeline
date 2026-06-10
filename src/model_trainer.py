import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split

def retrain_production_model(data_path, model_output_path):
    """
    Automated pipeline to retrain the LightGBM model on updated data.
    """
    print(f"Loading updated dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Preprocessing
    df = df.dropna(subset=['is_churn'])
    df = df.fillna(0)
    
    X = df.drop(columns=['msno', 'is_churn'])
    y = df['is_churn']
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Initiating LightGBM training sequence...")
    # Training with balanced class weights
    model = lgb.LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # Save the updated model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Updated production model saved to {model_output_path}")

if __name__ == "__main__":
    # If run directly from the terminal, execute the retraining pipeline
    INPUT_DATA = "data/processed/master_churn_data.parquet"
    OUTPUT_MODEL = "models/lightgbm_churn_model.pkl"
    
    if os.path.exists(INPUT_DATA):
        retrain_production_model(INPUT_DATA, OUTPUT_MODEL)
    else:
        print("Data file not found. Please run the DuckDB ETL pipeline first.")