import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Load model from MLflow Model Registry or local folder
MODEL_PATH = "models/wine_model"  # local path
# or from MLflow Registry:
# MODEL_PATH = "models:/wine-classifier/Production"

print("🚀 Loading model from:", MODEL_PATH)
model = mlflow.pyfunc.load_model(MODEL_PATH)

app = FastAPI(title="Wine Quality Prediction API")

# Define the input schema
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
def predict(data: WineFeatures):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
