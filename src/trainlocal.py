import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# Setup MLflow
# -------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-Classification-CSV")

# -------------------------------
# Load dataset from CSV
# -------------------------------
data_path = "../data/wine_dataset.csv"
df = pd.read_csv(data_path)
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train Model and Log
# -------------------------------
with mlflow.start_run() as run:
    n_estimators = 50
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log confusion matrix as an image
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log dataset as artifact
    mlflow.log_artifact(data_path)

    # Log model
    mlflow.sklearn.log_model(clf, "wine_model")

    print(f"✅ Run complete! Accuracy={acc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    #print(f"📂 Dataset and confusion matrix logged to MLflow.")

