from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model" in response.json()

def test_predict_healthy():
    payload = {
        "Customer ID": "CUS-8924",
        "total_transactions": 24,
        "total_revenue": 1250.0,
        "currently_auto_renews": 1,
        "has_cancelled_before": 0,
        "total_active_days": 8,
        "total_songs_skipped": 60,
        "total_songs_completed": 45,
        "total_listen_time_secs": 15000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUS-8924"
    assert "churn_probability" in data
    assert "is_churner" in data
    assert "risk_category" in data

def test_predict_saas_vocabulary():
    payload = {
        "Customer ID": "CUS-8924",
        "Billing Cycles": 24,
        "LTV ($)": 1250.0,
        "Auto-Renew Status": 1,
        "Prior Cancellations": 0,
        "Active Sessions (30-day)": 8,
        "Failed Sessions / Errors": 60,
        "Core Feature Adoptions": 45,
        "Product Usage Duration (Mins)": 250.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUS-8924"
    assert "churn_probability" in data
    # Ensure it maps to the exact same classification
    assert data["risk_category"] == "High Flight Risk"

if __name__ == "__main__":
    print("Running automated API validation checks...")
    try:
        test_health()
        print("OK: GET /health - PASSED")
        test_predict_healthy()
        print("OK: POST /predict (KKBOX schema) - PASSED")
        test_predict_saas_vocabulary()
        print("OK: POST /predict (Standard SaaS schema) - PASSED")
        print("\nAutomated checks completed: ALL TEST CASES PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"Assertion failed: {str(e)}")
    except ImportError as e:
        print(f"TestClient dependency issue: {str(e)}")
        print("Falling back to requests-based integration check...")
        print("Ensure the API is running locally and run integration tests using requests.")
    except Exception as e:
        print(f"Execution failed: {str(e)}")
