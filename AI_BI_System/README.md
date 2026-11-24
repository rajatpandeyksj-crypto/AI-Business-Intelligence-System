# AI Business Intelligence System

A Streamlit web app that turns sales data into insights: KPIs, 6-month forecast, K-Means segmentation,
Isolation Forest anomaly alerts, and a natural-language "Ask your data" box.

## Run
```bash
pip install -r requirements.txt
python generate_sample_data.py     # optional: regenerates sample_data.csv
streamlit run app.py
```
Open http://localhost:8501. Upload your own CSV/Excel in the sidebar, or use the bundled sample data.

## Input format
Columns: `month` (date), `region`, `category`, `revenue`, `cost` (at least 12 months for forecasting).

## Files
| File | Purpose |
|---|---|
| `bi_engine.py` | cleaning, KPIs, forecasting, segmentation, anomaly detection, NL query |
| `app.py` | Streamlit dashboard (5 tabs) |
| `generate_sample_data.py` | creates `sample_data.csv` |
