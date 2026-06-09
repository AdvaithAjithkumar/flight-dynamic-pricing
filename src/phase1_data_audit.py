import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/raw/Clean_Dataset.csv")

# ── Basic Shape ───────────────────────────────────────────
print("=" * 50)
print("SHAPE")
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

# ── Column Names & Dtypes ─────────────────────────────────
print("\n" + "=" * 50)
print("COLUMNS & DTYPES")
print(df.dtypes)

# ── First 5 Rows ──────────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE ROWS")
print(df.head())

# ── Null Check ────────────────────────────────────────────
print("\n" + "=" * 50)
print("NULL VALUES")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "No nulls found.")

# ── Duplicates ────────────────────────────────────────────
print("\n" + "=" * 50)
print("DUPLICATES")
dupes = df.duplicated().sum()
print(f"{dupes:,} duplicate rows found.")

# ── Target Variable: price ────────────────────────────────
print("\n" + "=" * 50)
print("TARGET VARIABLE: price")
print(df["price"].describe())

# ── Cardinality of Categorical Columns ───────────────────
print("\n" + "=" * 50)
print("CATEGORICAL UNIQUE VALUES")
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"  {col}: {df[col].nunique()} unique → {df[col].unique()}")
    # ── Drop useless columns & save cleaned version ───────────
df.drop(columns=["Unnamed: 0", "flight"], inplace=True)

df.to_csv("data/Clean_Dataset_v1.csv", index=False)
print("\nCleaned dataset saved → data/Clean_Dataset_v1.csv")
print(f"Final shape: {df.shape}")