import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os

from xgboost import XGBRegressor

# Reload model using XGBoost native format to fix SHAP compatibility
model = XGBRegressor()
model.load_model("models/xgboost_tuned.json")
X_test   = pd.read_csv("data/X_test.csv")
y_test   = pd.read_csv("data/y_test.csv").squeeze()

os.makedirs("outputs/shap", exist_ok=True)

# ── Sample for SHAP (full 60K is slow) ───────────────────
# 2000 rows is enough for stable SHAP values
sample_idx = np.random.default_rng(42).choice(len(X_test), 2000, replace=False)
X_sample   = X_test.iloc[sample_idx].reset_index(drop=True)

print("Computing SHAP values (1-2 mins)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("✓ SHAP values computed")

# ══════════════════════════════════════════════════════════
# PLOT 1 — Beeswarm (global feature importance with direction)
# ══════════════════════════════════════════════════════════
print("Generating beeswarm plot...")
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP Beeswarm — Top 20 Features (Tuned XGBoost)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap/01_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 01_shap_beeswarm.png saved")

# ══════════════════════════════════════════════════════════
# PLOT 2 — Bar plot (mean absolute SHAP — clean ranking)
# ══════════════════════════════════════════════════════════
print("Generating bar plot...")
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap/02_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_shap_bar.png saved")

# ══════════════════════════════════════════════════════════
# PLOT 3 — Dependence plot: days_left (core dynamic signal)
# ══════════════════════════════════════════════════════════
print("Generating days_left dependence plot...")
plt.figure(figsize=(10, 6))
shap.plots.scatter(
    shap_values[:, "days_left"],
    color=shap_values[:, "class_encoded"],
    show=False
)
plt.title("SHAP Dependence — days_left (colored by class)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap/03_shap_dependence_days_left.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 03_shap_dependence_days_left.png saved")

# ══════════════════════════════════════════════════════════
# PLOT 4 — Dependence plot: class_encoded
# ══════════════════════════════════════════════════════════
print("Generating class dependence plot...")
plt.figure(figsize=(10, 6))
shap.plots.scatter(
    shap_values[:, "class_encoded"],
    color=shap_values[:, "days_left"],
    show=False
)
plt.title("SHAP Dependence — class_encoded (colored by days_left)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap/04_shap_dependence_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 04_shap_dependence_class.png saved")

# ══════════════════════════════════════════════════════════
# PLOT 5 — Waterfall: single prediction explained
#           Pick one Economy last-minute and one Business early
# ══════════════════════════════════════════════════════════
print("Generating waterfall plots...")

# Find a last-minute Economy flight in sample
lm_eco = X_sample[
    (X_sample["class_encoded"] == 0) & (X_sample["days_left"] <= 5)
].index
if len(lm_eco) > 0:
    idx = lm_eco[0]
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title("Waterfall — Last-Minute Economy Flight", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/shap/05a_waterfall_economy_lastminute.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 05a_waterfall_economy_lastminute.png saved")

# Find a Business early-bird flight in sample
bus_early = X_sample[
    (X_sample["class_encoded"] == 1) & (X_sample["days_left"] >= 40)
].index
if len(bus_early) > 0:
    idx = bus_early[0]
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title("Waterfall — Early-Bird Business Flight", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/shap/05b_waterfall_business_early.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 05b_waterfall_business_early.png saved")

# ══════════════════════════════════════════════════════════
# PRINT — Top 10 features by mean |SHAP|
# ══════════════════════════════════════════════════════════
shap_df = pd.DataFrame({
    "feature":    X_sample.columns,
    "mean_shap":  np.abs(shap_values.values).mean(axis=0)
}).sort_values("mean_shap", ascending=False)

print("\n" + "=" * 50)
print("TOP 10 FEATURES BY MEAN |SHAP|")
print(shap_df.head(10).to_string(index=False))
print("\nSHAP analysis complete. Check outputs/shap/ for all plots.")