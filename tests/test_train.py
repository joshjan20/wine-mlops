import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Path to your dataset (DVC-pulled or local)
DATA_PATH = "data/wine_dataset.csv"

@pytest.fixture
def load_data():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def test_train_model(load_data):
    X, y = load_data
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    
    # Check accuracy is reasonable (this is a simple sanity check)
    acc = accuracy_score(y_test, preds)
    assert acc > 0, "Accuracy should be greater than 0"

def test_model_predict_shape(load_data):
    X, y = load_data
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y), "Predictions length should match target length"
