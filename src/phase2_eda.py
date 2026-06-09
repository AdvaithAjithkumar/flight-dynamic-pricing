import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────
df = pd.read_csv("data/Clean_Dataset_v1.csv")
os.makedirs("outputs/eda", exist_ok=True)
sns.set_theme(style="darkgrid")

# ── 1. Price Distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price"], bins=100, color="steelblue", edgecolor="none")
axes[0].set_title("Price Distribution (Raw)")
axes[0].set_xlabel("Price (₹)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(df["price"]), bins=100, color="coral", edgecolor="none")
axes[1].set_title("Price Distribution (Log-transformed)")
axes[1].set_xlabel("log(Price)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/eda/01_price_distribution.png", dpi=150)
plt.close()
print("✓ 01_price_distribution.png saved")

# ── 2. Price by Class ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="class", y="price", palette=["steelblue", "coral"], ax=ax)
ax.set_title("Price by Class")
ax.set_xlabel("Class")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("outputs/eda/02_price_by_class.png", dpi=150)
plt.close()
print("✓ 02_price_by_class.png saved")

# ── 3. Price by Airline ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
order = df.groupby("airline")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="airline", y="price", order=order, palette="Set2", ax=ax)
ax.set_title("Price by Airline (sorted by median)")
ax.set_xlabel("Airline")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("outputs/eda/03_price_by_airline.png", dpi=150)
plt.close()
print("✓ 03_price_by_airline.png saved")

# ── 4. Price by Stops ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
stop_order = ["zero", "one", "two_or_more"]
sns.boxplot(data=df, x="stops", y="price", order=stop_order, palette="Set1", ax=ax)
ax.set_title("Price by Number of Stops")
ax.set_xlabel("Stops")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("outputs/eda/04_price_by_stops.png", dpi=150)
plt.close()
print("✓ 04_price_by_stops.png saved")

# ── 5. Price vs Days Left (THE KEY DYNAMIC PRICING PLOT) ──
fig, ax = plt.subplots(figsize=(14, 5))
daily_avg = df.groupby("days_left")["price"].median().reset_index()
ax.plot(daily_avg["days_left"], daily_avg["price"], color="steelblue", linewidth=2)
ax.set_title("Median Price vs Days Left Before Departure (Dynamic Pricing Signal)")
ax.set_xlabel("Days Left")
ax.set_ylabel("Median Price (₹)")
ax.invert_xaxis()  # left = far out, right = last minute
plt.tight_layout()
plt.savefig("outputs/eda/05_price_vs_days_left.png", dpi=150)
plt.close()
print("✓ 05_price_vs_days_left.png saved")

# ── 6. Price by Departure Time ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
time_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
sns.boxplot(data=df, x="departure_time", y="price", order=time_order, palette="coolwarm", ax=ax)
ax.set_title("Price by Departure Time")
ax.set_xlabel("Departure Time")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("outputs/eda/06_price_by_departure_time.png", dpi=150)
plt.close()
print("✓ 06_price_by_departure_time.png saved")

# ── 7. Price vs Duration ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sample = df.sample(5000, random_state=42)  # sample to avoid overplotting
ax.scatter(sample["duration"], sample["price"], alpha=0.3, color="steelblue", s=10)
ax.set_title("Price vs Flight Duration (5K sample)")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("outputs/eda/07_price_vs_duration.png", dpi=150)
plt.close()
print("✓ 07_price_vs_duration.png saved")

# ── 8. Correlation Heatmap (Numeric Only) ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
numeric_cols = df[["duration", "days_left", "price"]]
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("outputs/eda/08_correlation_heatmap.png", dpi=150)
plt.close()
print("✓ 08_correlation_heatmap.png saved")

# ── 9. Summary Stats by Class ─────────────────────────────
print("\n" + "=" * 50)
print("PRICE SUMMARY BY CLASS")
print(df.groupby("class")["price"].describe().round(0))

# ── 10. Flight Count by Airline ───────────────────────────
print("\n" + "=" * 50)
print("FLIGHT COUNT BY AIRLINE")
print(df["airline"].value_counts())
print("\n EDA complete. Check outputs/eda/ for all plots.")