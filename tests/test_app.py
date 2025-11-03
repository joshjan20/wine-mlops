import json
import requests

def test_predict():
    with open("tests/test_payload.json") as f:
        data = json.load(f)
    
    response = requests.post("http://localhost:8000/predict", json=data)
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    print("✅ API test passed, prediction:", result["prediction"])
