import pandas as pd
import numpy as np
import os

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/Clean_Dataset_v1.csv")
print(f"Loaded: {df.shape}")

# ══════════════════════════════════════════════════════════
# 1. LOG-TRANSFORM TARGET VARIABLE
# ══════════════════════════════════════════════════════════
df["price_log"] = np.log1p(df["price"])
print("✓ price_log created")

# ══════════════════════════════════════════════════════════
# 2. CLASS — Binary encode (Economy=0, Business=1)
# ══════════════════════════════════════════════════════════
df["class_encoded"] = (df["class"] == "Business").astype(int)
print("✓ class_encoded created  [Economy=0, Business=1]")

# ══════════════════════════════════════════════════════════
# 3. STOPS — Ordinal encode (zero=0, one=1, two_or_more=2)
# ══════════════════════════════════════════════════════════
stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
df["stops_encoded"] = df["stops"].map(stops_map)
print("✓ stops_encoded created  [zero=0, one=1, two_or_more=2]")

# ══════════════════════════════════════════════════════════
# 4. DEPARTURE TIME — Ordinal encode by price rank
#    (Late_Night=0 cheapest → Night=5 most expensive)
# ══════════════════════════════════════════════════════════
departure_time_map = {
    "Late_Night":    0,
    "Afternoon":     1,
    "Morning":       2,
    "Evening":       3,
    "Early_Morning": 4,
    "Night":         5
}
df["departure_time_encoded"] = df["departure_time"].map(departure_time_map)
print("✓ departure_time_encoded created  [price-rank ordered]")

# ══════════════════════════════════════════════════════════
# 5. ARRIVAL TIME — Ordinal encode by price rank
#    (same logic as departure time)
# ══════════════════════════════════════════════════════════
arrival_time_map = {
    "Late_Night":    0,
    "Afternoon":     1,
    "Morning":       2,
    "Evening":       3,
    "Early_Morning": 4,
    "Night":         5
}
df["arrival_time_encoded"] = df["arrival_time"].map(arrival_time_map)
print("✓ arrival_time_encoded created  [price-rank ordered]")

# ══════════════════════════════════════════════════════════
# 6. AIRLINE — One-hot encode
# ══════════════════════════════════════════════════════════
airline_dummies = pd.get_dummies(df["airline"], prefix="airline", drop_first=True)
df = pd.concat([df, airline_dummies], axis=1)
print(f"✓ airline one-hot encoded  {list(airline_dummies.columns)}")

# ══════════════════════════════════════════════════════════
# 7. SOURCE & DESTINATION CITY — One-hot encode
# ══════════════════════════════════════════════════════════
source_dummies = pd.get_dummies(df["source_city"], prefix="src", drop_first=True)
dest_dummies   = pd.get_dummies(df["destination_city"], prefix="dst", drop_first=True)
df = pd.concat([df, source_dummies, dest_dummies], axis=1)
print(f"✓ source_city one-hot encoded  {list(source_dummies.columns)}")
print(f"✓ destination_city one-hot encoded  {list(dest_dummies.columns)}")

# ══════════════════════════════════════════════════════════
# 8. BOOKING URGENCY — Non-linear days_left bucketing
#    Based on EDA insight: price spikes below 17 days
# ══════════════════════════════════════════════════════════
def booking_urgency(days):
    if days >= 50:
        return 0   # early bird — stable low prices
    elif days >= 17:
        return 1   # medium — slight climb
    else:
        return 2   # last minute — spike zone

df["booking_urgency"] = df["days_left"].apply(booking_urgency)
print("✓ booking_urgency created  [0=early, 1=medium, 2=last_minute]")

# ══════════════════════════════════════════════════════════
# 9. PRICE PER HOUR — Duration interaction with class
#    Captures that Business duration relationship differs
# ══════════════════════════════════════════════════════════
df["price_per_hour"] = df["price"] / df["duration"].replace(0, np.nan)
print("✓ price_per_hour created")

# ══════════════════════════════════════════════════════════
# 10. ROUTE — source + destination combined
# ══════════════════════════════════════════════════════════
df["route"] = df["source_city"] + "_to_" + df["destination_city"]
route_dummies = pd.get_dummies(df["route"], prefix="route", drop_first=True)
df = pd.concat([df, route_dummies], axis=1)
print(f"✓ route one-hot encoded  ({route_dummies.shape[1]} route columns)")

# ══════════════════════════════════════════════════════════
# 11. DROP ORIGINAL RAW COLUMNS (replaced by encoded ones)
# ══════════════════════════════════════════════════════════
drop_cols = [
    "airline", "source_city", "destination_city",
    "departure_time", "arrival_time", "stops",
    "class", "route", "price"        # keep price_log as target
]
df.drop(columns=drop_cols, inplace=True)
print(f"✓ Raw columns dropped")

# ══════════════════════════════════════════════════════════
# 12. FINAL CHECK
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("FINAL FEATURE SET")
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nNull check:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSample:\n{df.head(3)}")

# ══════════════════════════════════════════════════════════
# 13. SAVE
# ══════════════════════════════════════════════════════════
os.makedirs("data", exist_ok=True)
df.to_csv("data/Clean_Dataset_v2.csv", index=False)
print("\n Feature engineered dataset saved → data/Clean_Dataset_v2.csv")