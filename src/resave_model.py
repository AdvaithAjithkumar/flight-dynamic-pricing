import joblib
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Reload and retrain on best params then save as JSON
best_params = joblib.load("models/best_params.pkl")

df = pd.read_csv("data/Clean_Dataset_v2.csv")
X = df.drop(columns=["price_log", "price_per_hour"])
y = df["price_log"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(**best_params, n_jobs=-1, random_state=42, verbosity=0)
model.fit(X_train, y_train)
model.save_model("models/xgboost_tuned.json")
print("Model re-saved as JSON → models/xgboost_tuned.json")