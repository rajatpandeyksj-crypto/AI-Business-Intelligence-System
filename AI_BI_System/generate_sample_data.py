"""Generate sample_data.csv (24 months x 4 regions x 4 categories) with a few planted anomalies."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
months = pd.date_range("2024-01-01", periods=24, freq="MS")
base = {"Electronics": 90000, "Apparel": 55000, "Grocery": 70000, "Home": 40000}
reg = {"North": 1.1, "South": 0.9, "East": 1.0, "West": 1.2}
rows = []
for i, m in enumerate(months):
    season = 1 + 0.25 * np.sin(2 * np.pi * (m.month - 1) / 12)
    for r in reg:
        for c in base:
            rev = base[c] * reg[r] * season * (1 + 0.012 * i) * rng.normal(1, 0.05)
            rows.append([m.date(), r, c, round(rev, 2), round(rev * rng.uniform(0.58, 0.72), 2)])
df = pd.DataFrame(rows, columns=["month", "region", "category", "revenue", "cost"])
df.loc[[37, 150, 301], "revenue"] *= [2.6, 0.35, 2.4]       # planted anomalies
df.loc[[10, 77], "revenue"] = np.nan                         # missing values
df.to_csv("sample_data.csv", index=False)
print("sample_data.csv written:", len(df), "rows")
