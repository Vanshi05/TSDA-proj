"""Streamlit frontend for the Indian Market Risk & Forecast Desk. Talks to the FastAPI backend."""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API = os.environ.get("TSDA_API", "http://127.0.0.1:8000")
LABELS = {"NIFTY50": "NIFTY 50", "BANK": "NIFTY Bank", "IT": "NIFTY IT", "PHARMA": "NIFTY Pharma", "VIX": "India VIX"}

st.set_page_config(page_title="Indian Market Risk & Forecast Desk", page_icon="📈", layout="wide")


@st.cache_data(ttl=600, show_spinner=False)
def api(path: str, **params):
    params = {k: v for k, v in params.items() if v not in (None, "")}
    r = requests.get(f"{API}{path}", params=params, timeout=600)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Market Risk & Forecast Desk")
try:
    requests.get(f"{API}/health", timeout=5).raise_for_status()
except Exception:
    st.error(f"Backend not reachable at {API}. Start it with `./run_backend.sh`.")
    st.stop()

asset = st.sidebar.selectbox("Index", list(LABELS), format_func=LABELS.get)
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("From", pd.Timestamp("2015-01-01"), min_value=pd.Timestamp("2010-01-01")).isoformat()
end = c2.date_input("To", pd.Timestamp.today()).isoformat()
st.sidebar.caption("Analysis uses sessions inside this range (min. 400).")
target_vol = st.sidebar.slider("Target portfolio volatility (% annual)", 5, 30, 12)
st.sidebar.markdown("---")
st.sidebar.caption("Educational decision-support tool. Not investment advice.")

st.title("Indian Market Risk & Forecast Desk")
st.caption("Forecasts, volatility risk and anomaly alerts for NIFTY indices. Built on ARIMA, Holt, GRU, GARCH and robust anomaly detection.")

tabs = st.tabs(["Dashboard & alerts", "Data quality & EDA", "Forecast", "Volatility & risk", "Anomalies", "About & limits"])

try:
    # ------------------------------------------------------------ dashboard
    with tabs[0]:
        s = api("/series", asset=asset, start=start, end=end)
        al = api("/alerts", asset=asset, start=start, end=end, target_vol=target_vol)
        sm = s["summary"]
        m = st.columns(5)
        m[0].metric(f"{LABELS[asset]} close", f"{sm['last']:,.2f}", f"{sm['change_1d_pct']:+.2f}% 1d")
        m[1].metric("1-month", f"{sm['change_1m_pct']:+.2f}%")
        m[2].metric("1-year", f"{sm['change_1y_pct']:+.2f}%")
        m[3].metric("Drawdown from peak", f"{sm['drawdown_pct']:.1f}%")
        m[4].metric("Forecast vol (ann.)", f"{al['annualised_vol_pct']:.1f}%")
        st.caption(f"Latest session in data: {sm['last_date']}")

        st.subheader("Alerts")
        icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}
        for a in al["alerts"]:
            (st.error if a["level"] == "high" else st.warning if a["level"] == "medium" else st.success)(f"{icon[a['level']]} {a['text']}")

        st.subheader("Suggested risk stance")
        r = st.columns(4)
        r[0].metric("1-day 95% VaR", f"{al['var95_pct']:.2f}%", help="Loss exceeded on ~1 in 20 days, per GARCH-t")
        r[1].metric("1-day 99% VaR", f"{al['var99_pct']:.2f}%")
        r[2].metric("India VIX", f"{al['vix']:.1f}", f"pct. rank {al['vix_percentile']:.0f}", delta_color="off")
        mult = al["position_multiplier"]
        r[3].metric(f"Exposure vs. normal @ {target_vol}% target", f"{mult * 100:.0f}%",
                    help="Volatility targeting: target vol ÷ forecast vol, capped at 200%.")
        if mult < 0.7:
            st.info(f"Forecast volatility is well above your {target_vol}% target: volatility targeting suggests running about {mult * 100:.0f}% of your usual exposure.")
        elif mult > 1.2:
            st.info(f"Forecast volatility is below your target: exposure up to {mult * 100:.0f}% of normal would keep risk on target.")
        else:
            st.info(f"Forecast volatility is close to your target; around {mult * 100:.0f}% of normal exposure keeps risk on target.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05)
        fig.add_scatter(x=s["dates"], y=s["close"], name="Close", line=dict(width=1.3), row=1, col=1)
        fig.add_scatter(x=s["dates"], y=s["ma20"], name="20d mean", line=dict(width=1), row=1, col=1)
        fig.add_scatter(x=s["dates"], y=s["ma60"], name="60d mean", line=dict(width=1), row=1, col=1)
        fig.add_scatter(x=s["dates"], y=s["rolling_vol_ann"], name="20d realised vol (%)", line=dict(width=1, color="#c0392b"), row=2, col=1)
        fig.update_layout(height=520, margin=dict(t=20, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ EDA
    with tabs[1]:
        q = api("/quality", start=start)
        st.subheader("Data source & quality")
        st.markdown("Source: community-maintained [Indian Market Historical Data](https://github.com/aswinv90/indian-market-historical-data) "
                    "(daily index closes, Parquet). Cleaning: invalid dates dropped, duplicates removed (keep last), non-positive/missing closes dropped, "
                    "series aligned on common trading dates (no interpolation). Extreme moves are flagged, not removed.")
        qdf = pd.DataFrame(q["reports"]).set_index("asset")
        st.dataframe(qdf, use_container_width=True)
        st.caption(f"Common aligned panel: {q['common_first']} to {q['common_last']} ({q['common_rows']:,} sessions; "
                   f"{q['rows_dropped_for_alignment']} rows dropped for alignment).")

        e = api("/eda", asset=asset, start=start, end=end)
        st.subheader(f"Distribution of {LABELS[asset]} daily returns")
        rs = e["return_stats"]
        k = st.columns(5)
        k[0].metric("Mean / day", f"{rs['mean_pct']:.3f}%")
        k[1].metric("Std / day", f"{rs['std_pct']:.2f}%")
        k[2].metric("Skewness", f"{rs['skew']:.2f}")
        k[3].metric("Excess kurtosis", f"{rs['excess_kurtosis']:.1f}")
        k[4].metric("Worst / best day", f"{rs['min_pct']:.1f}% / {rs['max_pct']:.1f}%")
        st.caption("Large excess kurtosis means fat tails: extreme days are far more common than a normal distribution implies, which is why we use Student-t GARCH for risk.")
        fh = go.Figure(go.Histogram(x=e["returns_pct"], nbinsx=120))
        fh.update_layout(height=300, margin=dict(t=10), xaxis_title="Daily log return (%)", yaxis_title="Days")
        st.plotly_chart(fh, use_container_width=True)

        st.subheader("Stationarity (ADF & KPSS)")
        sdf = pd.DataFrame(e["stationarity"]).set_index("series")
        st.dataframe(sdf.style.format({"adf_stat": "{:.2f}", "adf_p": "{:.4f}", "kpss_stat": "{:.3f}", "kpss_p": "{:.3f}"}), use_container_width=True)
        st.caption("ADF null = unit root; KPSS null = stationary. Price levels are non-stationary, returns are stationary, which justifies differencing (d=1) in ARIMA.")

        st.subheader("Autocorrelation")
        ac = e["acf"]
        fa = make_subplots(rows=1, cols=2, subplot_titles=["ACF of returns", "ACF of squared returns"])
        fa.add_bar(x=ac["lags"], y=ac["acf_returns"], row=1, col=1)
        fa.add_bar(x=ac["lags"], y=ac["acf_squared"], row=1, col=2)
        for col in (1, 2):
            for sign in (1, -1):
                fa.add_hline(y=sign * ac["conf"], line_dash="dot", line_color="grey", row=1, col=col)
        fa.update_layout(height=320, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fa, use_container_width=True)
        st.caption("Returns are almost uncorrelated (little linear predictability), but squared returns are strongly autocorrelated: volatility clusters, motivating GARCH.")

        st.subheader("Monthly returns heatmap")
        mdf = pd.DataFrame(e["monthly"]).pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
        fm = go.Figure(go.Heatmap(z=mdf.values, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                  y=mdf.index, colorscale="RdYlGn", zmid=0, colorbar=dict(title="%")))
        fm.update_layout(height=max(300, 28 * len(mdf)), margin=dict(t=10), yaxis=dict(autorange="reversed", type="category"))
        st.plotly_chart(fm, use_container_width=True)

        st.subheader("STL decomposition of absolute returns (period = 5 sessions)")
        stl = e["stl"]
        fs = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           subplot_titles=["Observed |r|", "Trend", "Weekly seasonal", "Residual"])
        for i, key in enumerate(["observed", "trend", "seasonal", "resid"], 1):
            fs.add_scatter(x=stl["dates"], y=stl[key], line=dict(width=0.8), row=i, col=1)
        fs.update_layout(height=620, showlegend=False, margin=dict(t=30))
        st.plotly_chart(fs, use_container_width=True)
        st.caption(f"Seasonal strength = {stl['seasonal_strength']:.2f} (0 = none, 1 = fully seasonal). "
                   "The trend component tracks volatility regimes; any weekly seasonality is weak.")

    # ------------------------------------------------------------- forecast
    with tabs[2]:
        st.subheader("Forecast the next sessions")
        h = st.slider("Horizon (trading days)", 1, 30, 10)
        fa_ = api("/forecast/ahead", asset=asset, start=start, end=end, horizon=h)
        hist = api("/series", asset=asset, start=start, end=end)
        n_hist = min(120, len(hist["dates"]))
        fg = go.Figure()
        fg.add_scatter(x=hist["dates"][-n_hist:], y=hist["close"][-n_hist:], name="History")
        fg.add_scatter(x=fa_["dates"] + fa_["dates"][::-1], y=fa_["upper95"] + fa_["lower95"][::-1], fill="toself",
                       fillcolor="rgba(52,152,219,0.15)", line=dict(width=0), name="95% interval")
        fg.add_scatter(x=fa_["dates"] + fa_["dates"][::-1], y=fa_["upper80"] + fa_["lower80"][::-1], fill="toself",
                       fillcolor="rgba(52,152,219,0.30)", line=dict(width=0), name="80% interval")
        fg.add_scatter(x=fa_["dates"], y=fa_["point"], name=f"ARIMA{tuple(fa_['arima_order'])} point forecast", line=dict(dash="dash"))
        fg.update_layout(height=420, margin=dict(t=10), legend=dict(orientation="h"))
        st.plotly_chart(fg, use_container_width=True)
        lo, hi = fa_["lower95"][-1], fa_["upper95"][-1]
        st.info(f"On {fa_['dates'][-1]}, {LABELS[asset]} is expected around {fa_['point'][-1]:,.0f}, with a 95% range of "
                f"{lo:,.0f} to {hi:,.0f} ({(lo / fa_['last_close'] - 1) * 100:+.1f}% to {(hi / fa_['last_close'] - 1) * 100:+.1f}% vs. last close). "
                "The centre line is nearly flat because index returns are close to unpredictable; the useful information is the width of the interval, driven by GARCH volatility.")

        st.subheader("Model comparison on a held-out period")
        cc = st.columns(3)
        test_size = cc[0].slider("Holdout (sessions)", 60, 400, 126, step=21)
        use_gru = cc[1].checkbox("Include GRU (slower)", value=True)
        if cc[2].button("Run comparison", type="primary") or st.session_state.get("ran_cmp"):
            st.session_state["ran_cmp"] = True
            with st.spinner("Walk-forward evaluation..."):
                cmp = api("/forecast/compare", asset=asset, start=start, end=end, test_size=test_size, include_gru=use_gru)
            mt = pd.DataFrame(cmp["metrics"]).T.astype(float)
            mt = mt.rename(columns={"RMSE_vs_naive_pct": "RMSE vs naive (%)", "DirAcc": "Direction acc. (%)", "Coverage95": "95% interval coverage (%)"})
            shown = mt.sort_values("RMSE").map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
            st.dataframe(shown, use_container_width=True)
            fc = go.Figure()
            fc.add_scatter(x=cmp["dates"], y=cmp["actual"], name="Actual", line=dict(width=2.5, color="black"))
            for name, vals in cmp["predictions"].items():
                fc.add_scatter(x=cmp["dates"], y=vals, name=name, line=dict(width=1.2), visible=True if name != "Holt (additive trend)" else "legendonly")
            fc.add_scatter(x=cmp["dates"] + cmp["dates"][::-1], y=cmp["interval_upper"] + cmp["interval_lower"][::-1], fill="toself",
                           fillcolor="rgba(52,152,219,0.12)", line=dict(width=0), name="Hybrid 95% interval")
            fc.update_layout(height=430, margin=dict(t=10), legend=dict(orientation="h"))
            st.plotly_chart(fc, use_container_width=True)
            best = min(cmp["metrics"], key=lambda k: cmp["metrics"][k]["RMSE"])
            naive_rmse = cmp["metrics"]["Naive (last close)"]["RMSE"]
            beaten = [k for k, v in cmp["metrics"].items() if v["RMSE"] < naive_rmse * 0.999]
            cov = cmp["metrics"]["Hybrid ARIMA+GARCH (point = ARIMA)"]["Coverage95"]
            st.markdown(
                f"**Interpretation.** Lowest RMSE: **{best}**. "
                + ("No model beats the naïve last-close benchmark by a meaningful margin: consistent with near-efficient markets, where tomorrow's price is best guessed by today's. "
                   if not beaten else f"Models beating naive: {', '.join(beaten)}. ")
                + f"The hybrid's 95% interval contained the actual close on **{cov:.0f}%** of holdout days, so its uncertainty estimate is "
                  f"{'well calibrated' if abs(cov - 95) <= 3 else 'miscalibrated'}. Direction accuracy near 50% means no real directional skill; "
                  "“n/a” means the model forecasts no change, so it makes no direction call.")

    # ----------------------------------------------------------------- risk
    with tabs[3]:
        rk = api("/risk", asset=asset, start=start, end=end, target_vol=target_vol)
        nd, bt = rk["next_day"], rk["backtest"]
        st.subheader("GARCH(1,1)-t volatility and Value-at-Risk")
        k = st.columns(4)
        k[0].metric("Next-day volatility", f"{nd['sigma_pct']:.2f}%")
        k[1].metric("Annualised", f"{nd['annualised_pct']:.1f}%", f"pct. rank {nd['percentile_vs_history']:.0f}", delta_color="off")
        k[2].metric("95% VaR (1 day)", f"{nd['var95_pct']:.2f}%")
        k[3].metric("99% VaR (1 day)", f"{nd['var99_pct']:.2f}%")
        fv = go.Figure()
        fv.add_scatter(x=rk["dates"], y=rk["actual_return_pct"], mode="lines", name="Daily return (%)", line=dict(width=0.7, color="grey"))
        fv.add_scatter(x=rk["dates"], y=[-v for v in rk["var95_pct"]], name="−95% VaR", line=dict(color="#e67e22", width=1.4))
        fv.add_scatter(x=rk["dates"], y=[-v for v in rk["var99_pct"]], name="−99% VaR", line=dict(color="#c0392b", width=1.4))
        fv.update_layout(height=380, margin=dict(t=10), legend=dict(orientation="h"), yaxis_title="%")
        st.plotly_chart(fv, use_container_width=True)
        st.subheader("Backtest on the last year")
        b = st.columns(4)
        b[0].metric("95% VaR breaches", f"{bt['var95_breaches']} / {bt['n']}", f"{bt['var95_breach_rate']:.1f}% (expect 5%)", delta_color="off")
        b[1].metric("99% VaR breaches", f"{bt['var99_breaches']} / {bt['n']}", f"{bt['var99_breach_rate']:.1f}% (expect 1%)", delta_color="off")
        b[2].metric("QLIKE: GARCH", f"{bt['qlike_garch']:.3f}")
        b[3].metric("QLIKE: constant vol", f"{bt['qlike_constant_vol']:.3f}")
        better = bt["qlike_garch"] < bt["qlike_constant_vol"]
        st.caption(f"GARCH {'beats' if better else 'does not beat'} the constant-volatility benchmark on QLIKE (lower is better). "
                   f"Persistence α+β = {rk['persistence']:.3f}: volatility shocks fade slowly.")
        fr = go.Figure()
        fr.add_scatter(x=rk["dates"], y=rk["garch_sigma_pct"], name="GARCH forecast σ")
        fr.add_scatter(x=rk["dates"], y=rk["realized_vol_20d"], name="20d realised σ")
        fr.update_layout(height=300, margin=dict(t=10), legend=dict(orientation="h"), yaxis_title="Daily σ (%)")
        st.plotly_chart(fr, use_container_width=True)
        with st.expander("Fitted GARCH parameters"):
            st.json(rk["params"])

    # ------------------------------------------------------------ anomalies
    with tabs[4]:
        st.subheader("Extreme-return detector")
        cc = st.columns(2)
        thr = cc[0].slider("Sensitivity: |robust z| threshold", 2.0, 8.0, 4.0, 0.5, help="Lower = more alerts")
        win = cc[1].slider("Look-back window (sessions)", 20, 250, 60, 10)
        an = api("/anomalies", asset=asset, start=start, end=end, window=win, threshold=thr)
        st.caption(f"{an['n_flagged']} of {an['n_obs']} sessions flagged. The median/MAD baseline uses only the previous {win} sessions, so a day cannot hide itself.")
        if an["recent_events"]:
            st.error(f"Recent anomalies (30 days): {', '.join(an['recent_events'])}")
        else:
            st.success("No anomalies in the last 30 days.")
        ev = pd.DataFrame(an["events"])
        fan = go.Figure()
        fan.add_scatter(x=an["dates"], y=an["return_pct"], mode="lines", name="Daily return (%)", line=dict(width=0.7))
        if not ev.empty:
            fan.add_scatter(x=ev["date"], y=ev["return_pct"], mode="markers", name="Flagged", marker=dict(color="red", size=7))
        fan.update_layout(height=380, margin=dict(t=10), yaxis_title="%")
        st.plotly_chart(fan, use_container_width=True)
        if not ev.empty:
            st.dataframe(ev.head(25).style.format({"return_pct": "{:.2f}", "robust_z": "{:.1f}"}), use_container_width=True)
            st.download_button("Download all flagged events (CSV)", ev.to_csv(index=False), f"{asset}_anomalies.csv", "text/csv")

    # ---------------------------------------------------------------- about
    with tabs[5]:
        st.markdown("""
### Problem
Retail investors, portfolio analysts and risk teams need a quick, honest view of *how risky the market is right now*, not a price prediction that
pretends to beat the market. This tool turns daily NIFTY index history into volatility forecasts, Value-at-Risk, forecast ranges and anomaly alerts.

### Method
| Component | Technique | Why |
|---|---|---|
| Stationarity | ADF + KPSS | Justify differencing |
| Dependence | ACF/PACF on returns & squared returns | Reveals volatility clustering |
| Seasonality | STL on absolute returns | Tests weekly pattern |
| Price forecasts | Naïve, ARIMA, Holt, GRU | Baseline vs classical vs ML |
| Hybrid | ARIMA mean + GARCH(1,1)-t variance | Calibrated forecast intervals |
| Risk | GARCH-t VaR, volatility targeting | Actionable exposure guidance |
| Anomalies | Rolling median/MAD robust z-score | Outlier-resistant, no look-ahead |

### Limitations
* Index returns are close to unpredictable; point forecasts rarely beat naïve. Treat the **interval and risk numbers**, not the centre line, as the product.
* Data are community-maintained; check coverage before relying on it.
* GARCH assumes stable parameters; regime breaks (e.g. March 2020) cause VaR breaches.
* No transaction costs, taxes or portfolio constraints; volatility targeting is a heuristic.
* Not investment advice.
""")
except RuntimeError as err:
    st.error(f"Backend error: {err}")
