"""Data retrieval, caching and quality handling for the Indian index panel."""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/aswinv90/indian-market-historical-data/main/data/INDICES/"
INDEX_FILES = {
    "NIFTY50": "NIFTY_50.parquet",
    "BANK": "NIFTY_BANK.parquet",
    "IT": "NIFTY_IT.parquet",
    "PHARMA": "NIFTY_PHARMA.parquet",
    "VIX": "INDIA_VIX.parquet",
}
INDEX_LABELS = {
    "NIFTY50": "NIFTY 50", "BANK": "NIFTY Bank", "IT": "NIFTY IT",
    "PHARMA": "NIFTY Pharma", "VIX": "India VIX",
}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_TTL_SECONDS = 24 * 3600


def _parse(raw: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), None)
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col])
        df.index = pd.DatetimeIndex(dates, name="Date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "Date"
    else:
        df = df.reset_index()
        cand = next((c for c in df.columns if str(c).lower() in {"index", "date", "datetime", "timestamp"}), None)
        if cand is None:
            raise ValueError(f"No date column in {filename}: {df.columns.tolist()}")
        dates = pd.to_datetime(df[cand], errors="coerce")
        df = df.drop(columns=[cand])
        df.index = pd.DatetimeIndex(dates, name="Date")
    close_col = {str(c).lower().replace("_", " ").strip(): c for c in df.columns}.get("close")
    if close_col is None:
        raise ValueError(f"No Close column in {filename}: {df.columns.tolist()}")
    return df.rename(columns={close_col: "Close"})


def _raw_bytes(filename: str) -> bytes:
    """Return parquet bytes, using a local cache (refreshed daily; stale cache used if offline)."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / filename
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS
    if fresh:
        return path.read_bytes()
    try:
        resp = requests.get(BASE_URL + filename, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return resp.content
    except requests.RequestException:
        if path.exists():
            return path.read_bytes()
        raise


def load_raw(asset: str) -> tuple[pd.DataFrame, dict]:
    """Load one index and clean it. Returns (clean frame, quality report)."""
    df = _parse(_raw_bytes(INDEX_FILES[asset]), INDEX_FILES[asset])
    report = {"asset": asset, "raw_rows": int(len(df))}
    report["invalid_dates"] = int(df.index.isna().sum())
    df = df.loc[~df.index.isna()].sort_index()
    report["duplicate_dates"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")]
    close = pd.to_numeric(df["Close"], errors="coerce")
    report["missing_close"] = int(close.isna().sum())
    report["nonpositive_close"] = int((close <= 0).sum())
    close = close.where(close > 0).dropna()
    # Extreme one-day moves are flagged (not removed): in market data they are usually genuine events.
    r = np.log(close).diff().dropna()
    report["extreme_moves_gt_10pct"] = int((r.abs() > 0.10).sum())
    report["first_date"] = str(close.index.min().date())
    report["last_date"] = str(close.index.max().date())
    report["clean_rows"] = int(len(close))
    return close.to_frame("Close"), report


def load_panel(start: str = "2010-01-01", end: str | None = None):
    """Common-date panel of closes plus per-series quality reports."""
    closes, reports = {}, {}
    for asset in INDEX_FILES:
        df, rep = load_raw(asset)
        closes[asset] = df["Close"]
        reports[asset] = rep
    panel = pd.concat(closes, axis=1).sort_index().loc[start:end]
    n_before = len(panel)
    panel = panel.dropna(how="any")
    return panel, reports, n_before - len(panel)
