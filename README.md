# Indian Market Risk & Forecast Desk

A TSDA mini-project: a working web application (FastAPI backend + Streamlit frontend) that turns daily NIFTY index data into
forecast ranges, GARCH volatility / Value-at-Risk, anomaly alerts and an exposure suggestion. The analysis notebook
(`time_series_financial_pipeline.ipynb`) documents the full methodology; the app uses the same methods on live data.

**Problem:** retail investors and risk analysts need a quick, honest view of how risky the market is *now*. Index prices are
close to unpredictable, so the app focuses on calibrated uncertainty (volatility, VaR, intervals) and unusual-move alerts rather than price-picking.

## Project layout

| Path | Purpose |
|---|---|
| `backend/data.py` | Download + local cache of Parquet data, cleaning and quality report |
| `backend/analysis.py` | Stationarity, ACF, STL, forecasting models, hybrid ARIMA+GARCH, VaR, anomaly detection |
| `backend/main.py` | FastAPI endpoints (`/series /quality /eda /forecast/compare /forecast/ahead /risk /anomalies /alerts`) |
| `frontend/app.py` | Streamlit dashboard (6 tabs) calling the backend |
| `time_series_financial_pipeline.ipynb` | Full analysis notebook (executed, outputs saved) |
| `REPORT.md` | Project report |
| `screenshots/` | Application screenshots |
| `outputs/` | CSV tables exported by the notebook |

## Data

Public Parquet files from [aswinv90/indian-market-historical-data](https://github.com/aswinv90/indian-market-historical-data):
NIFTY 50, NIFTY Bank, NIFTY IT, NIFTY Pharma, India VIX. Daily closes, 2007/2011 to present. Downloaded on first use and cached in
`data_cache/` (refreshed daily). Internet access is required for the first run.

## Setup

Python 3.10+ (tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the application

Terminal 1 (backend, http://127.0.0.1:8000, API docs at /docs):

```bash
./run_backend.sh
```

Terminal 2 (frontend, http://localhost:8501):

```bash
./run_frontend.sh
```

Set `TSDA_API` if the backend runs elsewhere. To run the notebook: `jupyter lab`, open the notebook, run all cells.

## Using the app

1. Pick an index and date range in the sidebar, and your target portfolio volatility.
2. **Dashboard & alerts**: headline metrics, alerts, VaR and a suggested exposure level.
3. **Data quality & EDA**: cleaning report, return distribution, ADF/KPSS, ACF, monthly heatmap, STL.
4. **Forecast**: 1-30 day fan chart; run the held-out model comparison (Naive, ARIMA, Holt, GRU, ARIMA+GARCH hybrid).
5. **Volatility & risk**: GARCH-t volatility, 95/99% VaR and a one-year backtest.
6. **Anomalies**: adjustable-sensitivity extreme-move detector with CSV download.

Not investment advice.
