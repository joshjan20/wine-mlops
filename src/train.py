import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# ===========================
#  CONFIGURATION
# ===========================
DATA_PATH = "data/wine_dataset.csv"       # DVC-tracked dataset
MODEL_DIR = "models/wine_model"           # Local model output
EXPERIMENT_NAME = "Wine-Classification"   # MLflow experiment name

# MLflow tracking URI (shared server or local)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT_NAME)

# ===========================
#  LOAD DATA
# ===========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run `dvc repro` or `create_dataset.py` first.")

df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
#  TRAIN & LOG MODEL
# ===========================
with mlflow.start_run() as run:
    # Hyperparameters
    n_estimators = 100
    max_depth = 6
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # ===========================
    #  LOG CONFUSION MATRIX
    # ===========================
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # ===========================
    #  SAVE MODEL LOCALLY
    # ===========================
    os.makedirs("models", exist_ok=True)
    mlflow.sklearn.save_model(model, MODEL_DIR)
    mlflow.sklearn.log_model(model, artifact_path="wine_model")

    # ===========================
    #  LOG DATASET (as artifact)
    # ===========================
    mlflow.log_artifact(DATA_PATH)

    print(f"✅ Training complete!")
    print(f"📊 Accuracy: {acc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(f"🔗 View run in MLflow UI: {mlflow.get_tracking_uri()}")

    mlflow.set_tracking_uri("http://23.22.232.131/:5000")  # replace with your EC2 public IP
    mlflow.set_experiment("Wine-Classification")