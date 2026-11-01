# Indian Market Risk & Forecast Desk — Project Report

*Time Series Data Analysis (TSDA) mini-project. Code, notebook and screenshots are in this repository; see the [README](../README.md) to run it.*

## 1. Problem and users
Equity-index prices are close to a random walk, so "predict tomorrow's NIFTY" is a poor product. What people can use is a reliable
answer to *how risky is the market now, how wide is the plausible range, and is today's move unusual?*
**Users:** retail investors who size positions, and student/junior risk analysts who monitor index risk.
**Objectives:** (1) a data-driven dashboard for NIFTY 50, Bank, IT, Pharma and India VIX; (2) compare forecasting methods against an honest
naïve benchmark; (3) forecast conditional volatility and Value-at-Risk; (4) flag extreme returns; (5) turn these into alerts and an exposure suggestion.

## 2. Data
Source: community-maintained repository *aswinv90/indian-market-historical-data* (Parquet). Daily closes; NIFTY 50/Bank/IT from 2007-09-17,
VIX from 2008-03-03, Pharma from 2011-01-31, all to 2026-10-23. Series are aligned on common trading dates
(2011-01-31 onward: 3,842 sessions; 2,890 sessions for the app's default 2015 start).
**Cleaning (implemented in `backend/data.py`, reported in the app):** invalid dates dropped, duplicate dates removed (keep last), missing and non-positive closes
dropped (one missing close per series found), no interpolation, extreme moves (>10%) flagged rather than removed (VIX has 297 because it is highly volatile;
index extremes such as 23-Mar-2020 are genuine events).

## 3. EDA findings
* Log price is non-stationary (ADF p = 0.84, KPSS rejects); log returns are stationary (ADF p ≈ 0, KPSS p ≥ 0.10). Hence ARIMA uses d = 1.
* Returns are fat-tailed and negatively skewed (2015+ NIFTY 50: skew −1.28, excess kurtosis 19.4), motivating Student-t errors.
* Returns show almost no autocorrelation, but squared returns do: volatility clustering.
* STL on absolute returns (period 5): the trend tracks volatility regimes (2020 spike); weekly seasonality is weak.

## 4. Methods and justification
| Method | Role | Why |
|---|---|---|
| Naïve (last close) | Benchmark | Any useful model must beat it |
| ARIMA(p,1,q), AIC grid on training data | Classical | Standard for non-stationary levels |
| Holt additive trend | Smoothing | Course syllabus; trend-following alternative |
| GRU (30-day window) | ML | Test whether non-linear sequence models add value |
| **Hybrid ARIMA + GARCH(1,1)-t** | Hybrid | ARIMA mean with GARCH variance gives calibrated prediction intervals |
| GARCH(1,1)-t VaR | Risk | Handles volatility clustering and fat tails |
| Rolling median/MAD robust z-score (window 60, threshold 4) | Anomaly | Outlier-resistant; uses only past data |
| VAR (5 indices, in notebook) | Cross-market dynamics | Syllabus; shows linkages |

All evaluation is chronological (no shuffling), the test period is never used for model selection or scaling, and forecasts are one-step walk-forward.

## 5. Results (from the executed notebook and the app)
**Price forecasts, last 252 sessions (notebook, 2025-10-14 to 2026-10-23):**

| Model | RMSE (points) | MAPE % |
|---|---|---|
| Naïve | 206.3 | 0.644 |
| ARIMA(0,1,0) | 206.3 | 0.644 |
| Holt | 207.1 | 0.647 |
| RNN | 343.3 | 1.137 |
| GRU | 362.7 | 1.207 |
| LSTM | 1076.3 | 3.790 |

**App run, 2015+ data, last 126 sessions:** ARIMA(1,1,0) RMSE 176.81 vs naïve 176.82, Holt 177.88, GRU 288.06. The hybrid's 95% interval covered **95.2%** of actual closes.
**Interpretation:** no model meaningfully beats the naïve benchmark; ARIMA(0,1,0) and Holt (α = 1, β = 0) collapse to "tomorrow = today", and the neural networks are
clearly worse in this sample. Direction accuracy is about 45–55%, i.e. no skill. This is consistent with near-efficient markets.
The valuable output is the *interval*, which is well calibrated.

**Volatility and risk (app, last 252 sessions):** GARCH persistence α+β ≈ 0.98 (notebook), so shocks decay slowly. 95% VaR was breached 18 times of 252 (7.1% vs 5% expected) and 99% VaR 3 times (1.2% vs 1%):
VaR is a little too optimistic at 95% in a turbulent year. GARCH QLIKE 0.508 beats constant-volatility 0.766.
**Anomalies:** 30 extreme days at |z| ≥ 4 in 2011–2026 (notebook), dominated by March 2020, plus 2015-08-24, 2024-06-04 and 2024-08-05.

**Latest state (as of 2026-10-23):** NIFTY 50 22,600, 14.2% below its peak in the 2015+ window; forecast annualised volatility 14.7% (percentile rank 68), 1-day 95% VaR 1.40%, 99% VaR 2.28%, India VIX 16.9; no anomaly in the last 30 days.

## 6. Application and decision support
The Streamlit frontend calls a FastAPI backend, so every chart and number is computed on request for the chosen index and date range. Use case (screenshots in [`screenshots/`](../screenshots/)):
an investor selects NIFTY 50 and a 12% target volatility; the dashboard shows forecast volatility 14.7%, 95% VaR 1.40% and suggests holding about 82% of normal
exposure (volatility targeting: target ÷ forecast vol, capped at 200%); alerts trigger for high volatility percentile, high VIX, deep drawdown or recent extreme moves.

## 7. Limitations and future work
* Price levels are barely predictable; the app does not claim to beat the market.
* VaR slightly underestimates risk at 95% in the last year; GARCH parameters assume stability and break at regime changes. Next steps: GJR-GARCH/EGARCH for leverage effects, filtered historical simulation.
* Community-maintained data, close prices only, 5-index universe; no transaction costs; volatility targeting is a heuristic.
* The directional-accuracy column is undefined (n/a) for flat forecasts; the notebook table shows 0.0 for the naïve and ARIMA(0,1,0) rows for that reason, whereas the app reports n/a.
* Future: CSV upload for any series, sector-VAR spillover alerts, scheduled alert emails, additional models (SARIMAX with VIX as exogenous input).

## 8. Reproducibility
See the [README](../README.md). Both the notebook and the app run end to end on the data above.
