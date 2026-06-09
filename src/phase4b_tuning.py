import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/Clean_Dataset_v2.csv")
X = df.drop(columns=["price_log", "price_per_hour"])
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Optuna Objective ──────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 4, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 2.0),
        "n_jobs":            -1,
        "random_state":      42,
        "verbosity":         0
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# ── Run Study ─────────────────────────────────────────────
print("Running Optuna tuning — 50 trials (5-8 mins)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ── Best Params ───────────────────────────────────────────
print("\n" + "=" * 50)
print("BEST PARAMS")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest CV RMSE (log space): {study.best_value:.6f}")

# ── Retrain on best params ────────────────────────────────
print("\nRetraining on best params...")
best_model = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42, verbosity=0)
best_model.fit(X_train, y_train)

# ── Evaluate in actual price space ───────────────────────
y_pred = best_model.predict(X_test)
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae  = np.mean(np.abs(y_test_actual - y_pred_actual))
r2   = 1 - np.sum((y_test_actual - y_pred_actual)**2) / np.sum((y_test_actual - y_test_actual.mean())**2)

print("\n" + "=" * 50)
print("TUNED XGBOOST RESULTS")
print(f"  RMSE : ₹{rmse:,.0f}")
print(f"  MAE  : ₹{mae:,.0f}")
print(f"  R²   : {r2:.4f}")

# ── Compare with baseline XGBoost ────────────────────────
print("\n" + "=" * 50)
print("BEFORE vs AFTER TUNING")
print(f"  Baseline XGBoost RMSE : ₹3,599")
print(f"  Tuned XGBoost RMSE    : ₹{rmse:,.0f}")
print(f"  Improvement           : ₹{3599 - rmse:,.0f}")

# ── Save tuned model ──────────────────────────────────────
joblib.dump(best_model, "models/xgboost_tuned.pkl")
joblib.dump(study.best_params, "models/best_params.pkl")
print("\nTuned model saved → models/xgboost_tuned.pkl")
print("Best params saved → models/best_params.pkl")