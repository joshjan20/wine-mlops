import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

mlflow.set_tracking_uri("http://23.22.232.131:5000")
EXPERIMENT_NAME = "Wine-Classification"

# Load model from MLflow Model Registry or local folder
#MODEL_PATH = "models/wine_model"  # local path
# or from MLflow Registry:
# MODEL_PATH = "models:/wine-classifier/Production"

#print("🚀 Loading model from:", MODEL_PATH)
#MODEL_PATH = "s3://mlflow-artifacts-john/wine_model"  # MLflow artifact location
#model = mlflow.pyfunc.load_model(MODEL_PATH)

# Load latest model dynamically
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise Exception(f"Experiment {EXPERIMENT_NAME} not found!")

# Get the latest run
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                          order_by=["start_time DESC"],
                          max_results=1)

if len(runs) == 0:
    raise Exception("No runs found in the experiment")

latest_run_id = runs.iloc[0]["run_id"]
MODEL_URI = f"runs:/{latest_run_id}/wine_model"

print("🚀 Loading latest model from:", MODEL_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

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
