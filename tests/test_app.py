import json
import pytest
from fastapi.testclient import TestClient
from src.app import app  # Make sure app.py exposes "app"

client = TestClient(app)

def test_predict_endpoint():
    with open("tests/test_payload.json") as f:
        data = json.load(f)

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], int)
    print("✅ FastAPI /predict test passed, prediction:", result["prediction"])
