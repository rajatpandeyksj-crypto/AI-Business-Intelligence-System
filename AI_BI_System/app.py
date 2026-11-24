"""AI Business Intelligence System - Streamlit dashboard.  Run: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import bi_engine as bi

st.set_page_config(page_title="AI Business Intelligence System", page_icon="📊", layout="wide")
st.title("📊 AI Business Intelligence System")
st.caption("Data → Insights → Forecast → Decisions")

# ---------- Sidebar: data ----------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    st.caption("Required columns: month, region, category, revenue, cost")
    horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
    contamination = st.slider("Anomaly sensitivity", 0.005, 0.05, 0.012, 0.001)

try:
    raw = bi.load_data(up if up else "sample_data.csv")
    df, report = bi.clean_data(raw)
    kpis = bi.compute_kpis(df)
    fc, mape = bi.forecast_sales(kpis["monthly"]["revenue"], horizon)
    seg = bi.segment_units(df)
    anomalies = bi.detect_anomalies(df, contamination)
except Exception as e:
    st.error(f"Could not process the data: {e}")
    st.stop()

with st.sidebar:
    st.success(f"{report['records']} records loaded" + ("" if up else " (sample data)"))
    st.caption(f"Cleaning: {report['missing_filled']} missing values filled, {report['duplicates_removed']} duplicates removed")
    regions = st.multiselect("Filter regions", sorted(df.region.unique()), default=sorted(df.region.unique()))
view = df[df.region.isin(regions)] if regions else df
vk = bi.compute_kpis(view)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast", "Segments", "Anomalies", "Ask your data"])
lakh = lambda v: f"Rs. {v/1e5:,.1f} L"

with tab1:
    c = st.columns(4)
    c[0].metric("Total Revenue", f"Rs. {vk['revenue']/1e7:,.2f} Cr")
    c[1].metric("Total Profit", f"Rs. {vk['profit']/1e7:,.2f} Cr")
    c[2].metric("Profit Margin", f"{vk['margin']:.1f}%")
    c[3].metric("YoY Growth", "n/a" if vk["growth"] is None else f"{vk['growth']:+.1f}%")
    m = vk["monthly"].reset_index()
    st.plotly_chart(px.line(m, x="month", y=["revenue", "profit"], title="Monthly Revenue & Profit",
                            labels={"value": "Rs.", "variable": ""}), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(vk["by_category"].reset_index(), x="category", y="revenue", title="Revenue by Category",
                          color="category"), width="stretch")
    b.plotly_chart(px.bar(vk["by_region"].reset_index(), x="region", y="revenue", title="Revenue by Region",
                          color="region"), width="stretch")

with tab2:
    st.subheader(f"{horizon}-month sales forecast")
    st.caption(f"Model: Linear Regression (trend + seasonality) · Hold-out MAPE: {mape:.2f}%")
    hist = kpis["monthly"].reset_index()
    fig = go.Figure()
    fig.add_scatter(x=hist.month, y=hist.revenue, name="Actual", mode="lines+markers")
    fig.add_scatter(x=fc.month, y=fc.forecast, name="Forecast", mode="lines+markers", line=dict(dash="dash", color="crimson"))
    fig.update_layout(yaxis_title="Revenue (Rs.)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fc.assign(month=fc.month.dt.strftime("%b %Y"), forecast=fc.forecast.round(0)), hide_index=True)
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "forecast.csv")

with tab3:
    st.subheader("Region × Category segments (K-Means, k=3)")
    st.plotly_chart(px.scatter(seg, x="revenue", y="margin_pct", color="segment", hover_data=["region", "category"],
                               size="revenue", labels={"margin_pct": "Margin %"}), width="stretch")
    st.dataframe(seg.sort_values("revenue", ascending=False).round(1), hide_index=True)

with tab4:
    st.subheader("Anomaly alerts (Isolation Forest)")
    if anomalies.empty: st.info("No anomalies detected.")
    else:
        for _, r in anomalies.iterrows():
            f = st.error if r.type == "SPIKE" else st.warning
            f(f"**{r.type}** · {r.month:%b %Y} · {r.region}-{r.category} · {lakh(r.revenue)} ({r.ratio:.2f}× normal)")
    st.caption("Ratio = revenue ÷ median of that region-category. Seasonal peaks can occasionally raise false alerts.")

with tab5:
    st.subheader("Ask your data")
    st.caption("Try: best region · worst category · forecast · any anomalies · profit margin · West")
    q = st.text_input("Your question")
    if q: st.info(bi.ask(q, df, kpis, fc, anomalies))
