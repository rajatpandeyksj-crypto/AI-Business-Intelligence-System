"""Core analytics engine for the AI Business Intelligence System."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

REQUIRED = ["month", "region", "category", "revenue", "cost"]


# ---------- 1. Load & clean ----------
def load_data(source) -> pd.DataFrame:
    """Read a CSV/Excel path or uploaded file object."""
    name = getattr(source, "name", str(source)).lower()
    df = pd.read_excel(source) if name.endswith((".xlsx", ".xls")) else pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {REQUIRED}")
    return df


def clean_data(df: pd.DataFrame):
    """Return (clean_df, report dict)."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])
    for c in ("revenue", "cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_missing = int(df[["revenue", "cost"]].isna().sum().sum())
    for c in ("revenue", "cost"):
        df[c] = df.groupby(["region", "category"])[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df["profit"] = df["revenue"] - df["cost"]
    return df.reset_index(drop=True), {"records": len(df), "missing_filled": n_missing, "duplicates_removed": n_dup}


# ---------- 2. KPIs ----------
def compute_kpis(df: pd.DataFrame) -> dict:
    monthly = df.groupby("month")[["revenue", "profit"]].sum()
    rev, prof = df["revenue"].sum(), df["profit"].sum()
    growth = None
    if len(monthly) >= 24:
        last, prev = monthly["revenue"].iloc[-12:].sum(), monthly["revenue"].iloc[-24:-12].sum()
        growth = (last / prev - 1) * 100
    elif len(monthly) >= 2:
        h = len(monthly) // 2
        growth = (monthly["revenue"].iloc[-h:].sum() / monthly["revenue"].iloc[:h].sum() - 1) * 100
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    by_reg = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    return {"revenue": rev, "profit": prof, "margin": prof / rev * 100, "growth": growth,
            "by_category": by_cat, "by_region": by_reg, "monthly": monthly}


# ---------- 3. Forecasting ----------
def _features(t):
    t = np.asarray(t)
    m = t % 12
    return np.column_stack([t, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])


def forecast_sales(monthly_revenue: pd.Series, horizon: int = 6):
    """Linear regression with trend + seasonality. Returns (forecast_df, mape)."""
    y = monthly_revenue.values
    n = len(y)
    if n < 12:
        raise ValueError("At least 12 months of data are needed for forecasting.")
    t = np.arange(n)
    split = max(int(n * 0.75), 6)
    mape = mean_absolute_percentage_error(
        y[split:], LinearRegression().fit(_features(t[:split]), y[:split]).predict(_features(t[split:]))) * 100
    model = LinearRegression().fit(_features(t), y)
    ft = np.arange(n, n + horizon)
    idx = pd.date_range(monthly_revenue.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.DataFrame({"month": idx, "forecast": model.predict(_features(ft))}), mape


# ---------- 4. Segmentation ----------
def segment_units(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    seg = df.groupby(["region", "category"]).agg(revenue=("revenue", "sum"), profit=("profit", "sum")).reset_index()
    seg["margin_pct"] = seg["profit"] / seg["revenue"] * 100
    k = min(k, len(seg))
    z = StandardScaler().fit_transform(seg[["revenue", "margin_pct"]])
    seg["cluster"] = KMeans(k, n_init=10, random_state=1).fit_predict(z)
    order = seg.groupby("cluster")["revenue"].mean().sort_values(ascending=False).index
    labels = ["High-Value", "Core", "Low-Value"][:k]
    seg["segment"] = seg["cluster"].map(dict(zip(order, labels)))
    return seg.drop(columns="cluster")


# ---------- 5. Anomaly detection ----------
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.012) -> pd.DataFrame:
    d = df.copy()
    d["ratio"] = d["revenue"] / d.groupby(["region", "category"])["revenue"].transform("median")
    flag = IsolationForest(contamination=contamination, random_state=1).fit_predict(d[["ratio"]]) == -1
    out = d[flag].copy()
    out["type"] = np.where(out["ratio"] > 1, "SPIKE", "DROP")
    return out.sort_values("ratio", ascending=False)[["month", "region", "category", "revenue", "ratio", "type"]]


# ---------- 6. Natural-language query ----------
def ask(question: str, df: pd.DataFrame, kpis: dict, fc: pd.DataFrame, anomalies: pd.DataFrame) -> str:
    q = question.lower()
    lakh = lambda v: f"Rs. {v / 1e5:,.2f} lakh"
    for r in kpis["by_region"].index:                       # revenue / profit of a named region or category
        if r.lower() in q:
            sub = df[df.region == r]
            return f"{r} region: revenue {lakh(sub.revenue.sum())}, profit {lakh(sub.profit.sum())}, margin {sub.profit.sum()/sub.revenue.sum()*100:.1f}%."
    for c in kpis["by_category"].index:
        if c.lower() in q:
            sub = df[df.category == c]
            return f"{c}: revenue {lakh(sub.revenue.sum())}, profit {lakh(sub.profit.sum())}, margin {sub.profit.sum()/sub.revenue.sum()*100:.1f}%."
    best = any(w in q for w in ("best", "top", "highest", "leading"))
    worst = any(w in q for w in ("worst", "lowest", "weakest", "least"))
    if "region" in q and (best or worst):
        s = kpis["by_region"]; n, v = (s.index[-1], s.iloc[-1]) if worst else (s.index[0], s.iloc[0])
        return f"{n} is the {'weakest' if worst else 'top'} region with {lakh(v)} revenue."
    if ("category" in q or "product" in q) and (best or worst):
        s = kpis["by_category"]; n, v = (s.index[-1], s.iloc[-1]) if worst else (s.index[0], s.iloc[0])
        return f"{n} is the {'weakest' if worst else 'top'} category with {lakh(v)} revenue."
    if "forecast" in q or "predict" in q or "next month" in q:
        return f"Forecast for {fc.month.iloc[0]:%b %Y}: {lakh(fc.forecast.iloc[0])}; next 6 months total {lakh(fc.forecast.sum())}."
    if "anomal" in q or "unusual" in q or "alert" in q:
        if anomalies.empty: return "No anomalies detected."
        a = anomalies.iloc[0]
        return f"{len(anomalies)} anomalies found. Largest: {a.type} in {a.region}-{a.category} ({a.month:%b %Y}), {a.ratio:.2f}x normal."
    if "margin" in q: return f"Overall profit margin is {kpis['margin']:.1f}%."
    if "profit" in q: return f"Total profit is {lakh(kpis['profit'])}."
    if "growth" in q or "yoy" in q:
        return "Not enough data for growth." if kpis["growth"] is None else f"Revenue growth is {kpis['growth']:+.1f}% year over year."
    if "revenue" in q or "sales" in q: return f"Total revenue is {lakh(kpis['revenue'])}."
    return ("Sorry, I couldn't understand that. Try: 'best region', 'worst category', 'forecast', "
            "'any anomalies', 'profit margin', or a region/category name like 'West'.")
