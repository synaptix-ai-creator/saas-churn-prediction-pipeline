import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Check paths
INPUT_DATA = "data/processed/master_churn_data.parquet"
MODEL_PATH = "models/lightgbm_churn_model.pkl"

if not os.path.exists(INPUT_DATA):
    print(f"Error: Dataset not found at {INPUT_DATA}")
    exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Trained model not found at {MODEL_PATH}")
    exit(1)

# Load data
print("Loading master dataset...")
df = pd.read_parquet(INPUT_DATA)

# Preprocessing to match trainer
df = df.dropna(subset=['is_churn'])
df = df.fillna(0)

# Split features
X = df.drop(columns=['msno', 'is_churn'])
y = df['is_churn']

# Stratified Split (same random state as trainer)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load production model
print("Loading LightGBM production model...")
model = joblib.load(MODEL_PATH)

# Run prediction
print("Running test set evaluations...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calibrated predictions
from src.data_processor import DataProcessor
processor = DataProcessor(MODEL_PATH.replace('lightgbm_churn_model.pkl', 'model_features.pkl'))

print("Applying calibration layer to test set...")
y_prob_calibrated = []
# Convert X_test to records for calibration
X_test_dict = X_test.to_dict(orient='records')
for idx, row_dict in enumerate(X_test_dict):
    raw_p = y_prob[idx] * 100.0
    cal_p = processor.calibrate_probability(row_dict, raw_p) / 100.0
    y_prob_calibrated.append(cal_p)
y_prob_calibrated = np.array(y_prob_calibrated)
y_pred_calibrated = np.where(y_prob_calibrated >= 0.5, 1, 0)

# Compute metrics for raw model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Compute metrics for calibrated model
accuracy_cal = accuracy_score(y_test, y_pred_calibrated)
precision_cal = precision_score(y_test, y_pred_calibrated)
recall_cal = recall_score(y_test, y_pred_calibrated)
f1_cal = f1_score(y_test, y_pred_calibrated)
auc_cal = roc_auc_score(y_test, y_prob_calibrated)

print("\n=== RAW MODEL PERFORMANCE METRICS ===")
print(f"Test Records Evaluated : {len(y_test):,}")
print(f"Accuracy Score         : {accuracy * 100:.2f}%")
print(f"Precision Score        : {precision * 100:.2f}%")
print(f"Recall Score (Catch)   : {recall * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {auc:.4f}")
print("=====================================")

print("\n=== CALIBRATED MODEL PERFORMANCE METRICS ===")
print(f"Test Records Evaluated : {len(y_test):,}")
print(f"Accuracy Score         : {accuracy_cal * 100:.2f}%")
print(f"Precision Score        : {precision_cal * 100:.2f}%")
print(f"Recall Score (Catch)   : {recall_cal * 100:.2f}%")
print(f"F1-Score               : {f1_cal * 100:.2f}%")
print(f"ROC-AUC Score          : {auc_cal:.4f}")
print("============================================\n")

print("Calibrated Model Classification Report:")
print(classification_report(y_test, y_pred_calibrated))

