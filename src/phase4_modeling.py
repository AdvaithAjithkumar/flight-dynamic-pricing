import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/Clean_Dataset_v2.csv")
print(f"Loaded: {df.shape}")

# ── Split features and target ─────────────────────────────
X = df.drop(columns=["price_log", "price_per_hour"])
y = df["price_log"]

# Note: price_per_hour dropped from X because it was computed
# from the original price — would be data leakage in the model

print(f"Features: {X.shape[1]}  |  Target: price_log")

# ── Train/Test Split (80/20) ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── Evaluation helper ─────────────────────────────────────
def evaluate(name, y_true, y_pred_log):
    # Convert predictions back from log space to actual price
    y_true_actual  = np.expm1(y_true)
    y_pred_actual  = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae  = mean_absolute_error(y_true_actual, y_pred_actual)
    r2   = r2_score(y_true_actual, y_pred_actual)

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  R²   : {r2:.4f}")
    return {"model": name, "RMSE": round(rmse), "MAE": round(mae), "R2": round(r2, 4)}

results = []

# ══════════════════════════════════════════════════════════
# MODEL 1 — Linear Regression (Baseline)
# ══════════════════════════════════════════════════════════
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append(evaluate("Linear Regression", y_test, y_pred_lr))

# ══════════════════════════════════════════════════════════
# MODEL 2 — Random Forest
# ══════════════════════════════════════════════════════════
print("\nTraining Random Forest (this will take 2-3 mins)...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, y_pred_rf))

# ══════════════════════════════════════════════════════════
# MODEL 3 — XGBoost
# ══════════════════════════════════════════════════════════
print("\n⏳ Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results.append(evaluate("XGBoost", y_test, y_pred_xgb))

# ── Comparison Table ──────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ── Save best model (XGBoost) ─────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgboost_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")
print("\nXGBoost model saved → models/xgboost_model.pkl")
print("Feature columns saved → models/feature_columns.pkl")

# ── Save test predictions for Phase 5 (SHAP) ─────────────
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("Test set saved → data/X_test.csv + data/y_test.csv")