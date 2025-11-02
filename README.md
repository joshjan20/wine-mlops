# Wine Classification ML Project

## Overview
This project demonstrates a **machine learning workflow** for classifying wines into three types (Class 0, Class 1, Class 2) based on their **chemical composition**. The model uses a **Random Forest Classifier** to predict the wine class and logs **metrics, hyperparameters, and model artifacts** using **MLflow**.

## Dataset
The project uses the **Wine dataset** from scikit-learn:
- **Features:** 13 numeric chemical properties of wine (e.g., alcohol, malic acid, color intensity)  
- **Target:** Wine class label (0, 1, or 2)

## Model
- Random Forest Classifier  
- Tracks metrics: **Accuracy, F1-score, Precision, Recall**  
- Lightweight and easy to run on CPU  

## Usage
1. Install dependencies:
```bash
pip install scikit-learn pandas numpy mlflow

