"""FastAPI backend exposing the TSDA analyses as JSON endpoints."""
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import analysis as A
from .data import INDEX_FILES, INDEX_LABELS, load_panel

app = FastAPI(title="Indian Market Risk & Forecast Desk API", version="1.0")

_panel_cache: dict = {}


def panel(start: str):
    key = (start, int(time.time() // 3600))  # refresh hourly
    if key not in _panel_cache:
        _panel_cache.clear()
        _panel_cache[key] = load_panel(start=start)
    return _panel_cache[key]


def series(asset: str, start: str, end: str | None) -> pd.Series:
    if asset not in INDEX_FILES:
        raise HTTPException(404, f"Unknown index '{asset}'")
    close = panel(start)[0][asset]
    if end:
        close = close.loc[:end]
    if len(close) < 400:
        raise HTTPException(422, "Selected range has too few observations (need at least 400 sessions).")
    return close


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/indices")
def indices():
    return [{"id": k, "label": INDEX_LABELS[k]} for k in INDEX_FILES]


@app.get("/series")
def get_series(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None):
    c = series(asset, start, end)
    r = log_returns(c)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in c.index], "close": c.tolist(),
        "ma20": c.rolling(20).mean().bfill().tolist(), "ma60": c.rolling(60).mean().bfill().tolist(),
        "rolling_vol_ann": (r.rolling(20).std() * np.sqrt(252) * 100).reindex(c.index).bfill().tolist(),
        "summary": {
            "last": float(c.iloc[-1]), "last_date": str(c.index[-1].date()),
            "change_1d_pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100),
            "change_1m_pct": float((c.iloc[-1] / c.iloc[-22] - 1) * 100),
            "change_1y_pct": float((c.iloc[-1] / c.iloc[-253] - 1) * 100),
            "drawdown_pct": float((c.iloc[-1] / c.cummax().iloc[-1] - 1) * 100),
            "n_obs": int(len(c)), "first_date": str(c.index[0].date()),
        },
    }


@app.get("/quality")
def quality(start: str = "2010-01-01"):
    p, reports, dropped = panel(start)
    return {"reports": list(reports.values()), "common_rows": int(len(p)),
            "rows_dropped_for_alignment": int(dropped),
            "common_first": str(p.index[0].date()), "common_last": str(p.index[-1].date())}


@app.get("/eda")
def eda(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None):
    c = series(asset, start, end)
    r = log_returns(c)
    from scipy import stats
    return {
        "stationarity": [A.stationarity(np.log(c), f"log({INDEX_LABELS[asset]})"),
                         A.stationarity(r, f"{INDEX_LABELS[asset]} log returns")],
        "acf": A.acf_table(r), "stl": A.stl_components(r),
        "return_stats": {"mean_pct": float(r.mean() * 100), "std_pct": float(r.std() * 100),
                         "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
                         "min_pct": float(r.min() * 100), "max_pct": float(r.max() * 100)},
        "returns_pct": (100 * r).tolist(),
        "monthly": _monthly(c),
    }


def _monthly(c: pd.Series):
    m = c.resample("ME").last().pct_change().dropna() * 100
    return [{"year": int(d.year), "month": int(d.month), "ret": float(v)} for d, v in m.items()]


@lru_cache(maxsize=16)
def _forecast_cached(asset, start, end, test_size, gru, panel_stamp):
    return A.compare_forecasts(series(asset, start, end), test_size=test_size, include_gru=gru)


@app.get("/forecast/compare")
def forecast_compare(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None,
                     test_size: int = Query(126, ge=30, le=500), include_gru: bool = True):
    stamp = int(time.time() // 86400)
    return _forecast_cached(asset, start, end, test_size, include_gru, stamp)


@lru_cache(maxsize=16)
def _ahead_cached(asset, start, end, horizon, stamp):
    return A.forecast_ahead(series(asset, start, end), horizon=horizon)


@app.get("/forecast/ahead")
def forecast_ahead(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None,
                   horizon: int = Query(10, ge=1, le=60)):
    return _ahead_cached(asset, start, end, horizon, int(time.time() // 86400))


@lru_cache(maxsize=16)
def _risk_cached(asset, start, end, test_size, target, stamp):
    return A.volatility_risk(log_returns(series(asset, start, end)), test_size=test_size, target_vol_ann=target)


@app.get("/risk")
def risk(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None,
         test_size: int = Query(252, ge=60, le=750), target_vol: float = Query(12.0, gt=1, le=60)):
    return _risk_cached(asset, start, end, test_size, target_vol, int(time.time() // 86400))


@app.get("/anomalies")
def anomalies(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None,
              window: int = Query(60, ge=20, le=250), threshold: float = Query(4.0, ge=2, le=10)):
    c = series(asset, start, end)
    df = A.detect_anomalies(log_returns(c), window, threshold).dropna(subset=["robust_z"])
    flagged = df[df["flag"]].copy()
    recent_cut = c.index[-1] - pd.Timedelta(days=30)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index], "return_pct": df["return_pct"].tolist(),
        "robust_z": df["robust_z"].tolist(),
        "events": [{"date": d.strftime("%Y-%m-%d"), "return_pct": float(row.return_pct), "robust_z": float(row.robust_z),
                    "direction": "drop" if row.return_pct < 0 else "spike"} for d, row in
                   flagged.sort_values("robust_z", key=lambda s: s.abs(), ascending=False).iterrows()],
        "recent_events": [d.strftime("%Y-%m-%d") for d in flagged.index if d >= recent_cut],
        "n_flagged": int(len(flagged)), "n_obs": int(len(df)),
    }


@app.get("/alerts")
def alerts(asset: str = "NIFTY50", start: str = "2010-01-01", end: str | None = None,
           target_vol: float = 12.0):
    """Decision-support summary combining volatility regime, VIX, drawdown and recent anomalies."""
    c = series(asset, start, end)
    stamp = int(time.time() // 86400)
    rk = _risk_cached(asset, start, end, 252, target_vol, stamp)
    nd = rk["next_day"]
    an = anomalies(asset, start, end, 60, 4.0)
    vix = panel(start)[0]["VIX"]
    vix_pct = float((vix < vix.iloc[-1]).mean() * 100)
    dd = float((c.iloc[-1] / c.cummax().iloc[-1] - 1) * 100)
    items = []
    if nd["percentile_vs_history"] >= 90:
        items.append(("high", f"Forecast volatility is in the top {100 - nd['percentile_vs_history']:.0f}% of its history "
                              f"({nd['annualised_pct']:.1f}% annualised). Consider reducing exposure."))
    elif nd["percentile_vs_history"] >= 70:
        items.append(("medium", f"Volatility is elevated ({nd['annualised_pct']:.1f}% annualised, "
                                f"percentile rank {nd['percentile_vs_history']:.0f}/100)."))
    else:
        items.append(("low", f"Volatility is calm ({nd['annualised_pct']:.1f}% annualised, "
                             f"percentile rank {nd['percentile_vs_history']:.0f}/100)."))
    if vix_pct >= 85:
        items.append(("high", f"India VIX at {vix.iloc[-1]:.1f} is above {vix_pct:.0f}% of its history: market is pricing stress."))
    if dd <= -10:
        items.append(("high" if dd <= -20 else "medium", f"{INDEX_LABELS[asset]} is {abs(dd):.1f}% below its peak in this window."))
    if an["recent_events"]:
        items.append(("high", f"Extreme return(s) in the last 30 days: {', '.join(an['recent_events'])}."))
    return {
        "asset": asset, "as_of": str(c.index[-1].date()), "alerts": [{"level": l, "text": t} for l, t in items],
        "var95_pct": nd["var95_pct"], "var99_pct": nd["var99_pct"],
        "position_multiplier": nd["position_multiplier"], "target_vol": target_vol,
        "vix": float(vix.iloc[-1]), "vix_percentile": vix_pct, "drawdown_pct": dd,
        "annualised_vol_pct": nd["annualised_pct"], "vol_percentile": nd["percentile_vs_history"],
    }
