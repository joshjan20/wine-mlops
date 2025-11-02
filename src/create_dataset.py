# src/create_dataset.py
from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save dataset to data folder
df.to_csv("../data/wine_dataset.csv", index=False)
print("✅ Dataset saved to ../data/wine_dataset.csv")

