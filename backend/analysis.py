"""Core TSDA computations: EDA, forecasting comparison, volatility/risk and anomaly detection."""
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- EDA
def stationarity(series: pd.Series, name: str) -> dict:
    x = series.dropna().astype(float)
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    adf_ok, kpss_ok = adf_p < 0.05, kpss_p >= 0.05
    verdict = ("stationary" if adf_ok and kpss_ok else
               "non-stationary" if not adf_ok and not kpss_ok else "inconclusive")
    return {"series": name, "adf_stat": float(adf_stat), "adf_p": float(adf_p),
            "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p), "verdict": verdict}


def acf_table(returns: pd.Series, nlags: int = 30) -> dict:
    n = len(returns)
    return {
        "lags": list(range(1, nlags + 1)),
        "acf_returns": acf(returns, nlags=nlags)[1:].tolist(),
        "acf_squared": acf(returns ** 2, nlags=nlags)[1:].tolist(),
        "conf": float(1.96 / np.sqrt(n)),
    }


def stl_components(returns: pd.Series, period: int = 5) -> dict:
    fit = STL(returns.abs(), period=period, robust=True).fit()
    seasonal_strength = max(0.0, 1 - float(np.var(fit.resid) / np.var(fit.seasonal + fit.resid)))
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in fit.trend.index],
        "observed": fit.observed.tolist(), "trend": fit.trend.tolist(),
        "seasonal": fit.seasonal.tolist(), "resid": fit.resid.tolist(),
        "seasonal_strength": seasonal_strength,
    }


# ------------------------------------------------------------ forecasting
def _metrics(actual_p, pred_p, actual_l, pred_l, prev_l) -> dict:
    mae = mean_absolute_error(actual_p, pred_p)
    rmse = float(np.sqrt(mean_squared_error(actual_p, pred_p)))
    mape = float(np.mean(np.abs((actual_p - pred_p) / actual_p)) * 100)
    pred_dir = np.sign(pred_l - prev_l)
    # A flat "no change" forecast carries no direction call, so accuracy is undefined (not 0%).
    dacc = None if not np.any(pred_dir) else float(np.mean(np.sign(actual_l - prev_l) == pred_dir) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape, "DirAcc": dacc}


def select_arima_order(train_log: np.ndarray, max_p: int = 2, max_q: int = 2):
    best = None
    for p, q in product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = ARIMA(train_log, order=(p, 1, q)).fit().aic
        except Exception:
            continue
        if best is None or aic < best[0]:
            best = (aic, (p, 1, q))
    if best is None:
        raise RuntimeError("No ARIMA candidate converged")
    return best[1]


def _gru_predict(nifty_log: pd.Series, split_idx: int, window: int = 30, epochs: int = 25):
    import torch
    from torch import nn
    torch.manual_seed(42)
    scaler = MinMaxScaler().fit(nifty_log.iloc[:split_idx].to_numpy().reshape(-1, 1))
    s = scaler.transform(nifty_log.to_numpy().reshape(-1, 1)).astype(np.float32).ravel()
    X = np.stack([s[i - window:i] for i in range(window, len(s))])[..., None]
    y = s[window:, None]
    cut_train = split_idx - window
    Xtr, ytr, Xte = X[:cut_train], y[:cut_train], X[cut_train:]
    v = int(len(Xtr) * 0.9)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn, self.head = nn.GRU(1, 32, batch_first=True), nn.Linear(32, 1)

        def forward(self, x):
            return self.head(self.rnn(x)[0][:, -1])

    net = Net()
    opt, loss_fn = torch.optim.Adam(net.parameters(), 1e-3), nn.MSELoss()
    Xa, ya = torch.tensor(Xtr[:v]), torch.tensor(ytr[:v])
    Xv, yv = torch.tensor(Xtr[v:]), torch.tensor(ytr[v:])
    best, state, wait = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(net(Xa[idx]), ya[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xv), yv))
        if vl < best - 1e-6:
            best, state, wait = vl, deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= 5:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte)).numpy()
    return scaler.inverse_transform(pred).ravel()


def compare_forecasts(close: pd.Series, test_size: int = 126, include_gru: bool = True) -> dict:
    """Walk-forward one-step comparison of Naive / ARIMA / Holt / GRU / ARIMA+GARCH hybrid."""
    log = np.log(close.astype(float))
    if len(log) < test_size + 250:
        raise ValueError("Not enough history for the chosen window and test size")
    split = len(log) - test_size
    train, test = log.iloc[:split], log.iloc[split:]
    prev = np.r_[train.iloc[-1], test.to_numpy()[:-1]]
    preds = {"Naive (last close)": prev.copy()}

    order = select_arima_order(train.to_numpy())
    res = ARIMA(train.to_numpy(), order=order).fit()
    ar = []
    for actual in test.to_numpy():
        ar.append(float(np.asarray(res.forecast(1))[0]))
        res = res.append([float(actual)], refit=False)
    ar = np.asarray(ar)
    preds[f"ARIMA{order}"] = ar

    holt = ExponentialSmoothing(train.to_numpy(), trend="add", initialization_method="estimated").fit(optimized=True)
    a, b = float(holt.params["smoothing_level"]), float(holt.params["smoothing_trend"])
    lvl, trd = float(np.asarray(holt.level)[-1]), float(np.asarray(holt.trend)[-1])
    hp = []
    for actual in test.to_numpy():
        hp.append(lvl + trd)
        new_l = a * actual + (1 - a) * (lvl + trd)
        trd = b * (new_l - lvl) + (1 - b) * trd
        lvl = new_l
    preds["Holt (additive trend)"] = np.asarray(hp)

    if include_gru:
        try:
            preds["GRU"] = _gru_predict(log, split)
        except ImportError:
            pass

    actual_p, actual_l = np.exp(test.to_numpy()), test.to_numpy()
    rows = {name: _metrics(actual_p, np.exp(p), actual_l, p, prev) for name, p in preds.items()}

    # Hybrid: ARIMA point forecast + GARCH one-step volatility -> 95% prediction interval.
    r_pct = 100 * log.diff().dropna()
    hybrid = hybrid_interval(r_pct, split_off=len(log) - split)
    sigma = hybrid["sigma_pct"] / 100
    lo, hi = np.exp(ar - 1.96 * sigma), np.exp(ar + 1.96 * sigma)
    coverage = float(np.mean((actual_p >= lo) & (actual_p <= hi)) * 100)
    rows["Hybrid ARIMA+GARCH (point = ARIMA)"] = {**rows[f"ARIMA{order}"], "Coverage95": coverage}
    naive_rmse = rows["Naive (last close)"]["RMSE"]
    for r in rows.values():
        r["RMSE_vs_naive_pct"] = float((r["RMSE"] / naive_rmse - 1) * 100)

    dates = [d.strftime("%Y-%m-%d") for d in test.index]
    return {
        "arima_order": list(order), "holt_alpha": a, "holt_beta": b,
        "metrics": rows, "dates": dates, "actual": actual_p.tolist(),
        "predictions": {k: np.exp(v).tolist() for k, v in preds.items()},
        "interval_lower": lo.tolist(), "interval_upper": hi.tolist(),
        "test_size": test_size,
    }


def hybrid_interval(r_pct: pd.Series, split_off: int) -> dict:
    """Sequential one-step GARCH(1,1)-t volatility over the last `split_off` returns (params fixed on train)."""
    from arch import arch_model
    n_test = split_off
    train = r_pct.iloc[:-n_test]
    test = r_pct.iloc[-n_test:]
    fit = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    om, al, be, mu = (float(fit.params[k]) for k in ("omega", "alpha[1]", "beta[1]", "mu"))
    var_prev = float(np.asarray(fit.conditional_volatility)[-1] ** 2)
    resid_prev = float(np.asarray(fit.resid)[-1])
    out = []
    for ret in test.to_numpy():
        v = max(om + al * resid_prev ** 2 + be * var_prev, 1e-12)
        out.append(np.sqrt(v))
        var_prev, resid_prev = v, float(ret - mu)
    return {"sigma_pct": np.asarray(out), "dates": test.index, "test_returns": test, "fit": fit,
            "mu": mu, "nu": float(fit.params["nu"])}


def forecast_ahead(close: pd.Series, horizon: int = 10, fit_years: int = 8) -> dict:
    """Multi-day price forecast: ARIMA mean + GARCH-t cumulative variance fan chart."""
    from arch import arch_model
    log = np.log(close.astype(float))
    log = log.iloc[-fit_years * 252:]
    order = select_arima_order(log.to_numpy())
    res = ARIMA(log.to_numpy(), order=order).fit()
    mean_path = np.asarray(res.forecast(horizon))
    r_pct = 100 * log.diff().dropna()
    g = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    var_path = np.asarray(g.forecast(horizon=horizon).variance.iloc[-1])
    cum_sd = np.sqrt(np.cumsum(var_path)) / 100
    last = float(close.iloc[-1])
    bdays = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    out = {"dates": [d.strftime("%Y-%m-%d") for d in bdays], "last_close": last,
           "arima_order": list(order), "point": np.exp(mean_path).tolist()}
    for level, z in (("80", 1.2816), ("95", 1.96)):
        out[f"lower{level}"] = np.exp(mean_path - z * cum_sd).tolist()
        out[f"upper{level}"] = np.exp(mean_path + z * cum_sd).tolist()
    return out


# ------------------------------------------------------- volatility / risk
def volatility_risk(returns: pd.Series, test_size: int = 252, target_vol_ann: float = 12.0) -> dict:
    r_pct = 100 * returns
    res = hybrid_interval(r_pct, split_off=test_size)
    sigma, test, nu, mu, fit = res["sigma_pct"], res["test_returns"], res["nu"], res["mu"], res["fit"]
    scale = np.sqrt((nu - 2) / nu)
    q95, q99 = stats.t.ppf(0.05, nu) * scale, stats.t.ppf(0.01, nu) * scale
    var95, var99 = -(mu + sigma * q95), -(mu + sigma * q99)
    loss = -test.to_numpy()
    fvar = sigma ** 2
    rv_proxy = test.to_numpy() ** 2
    qlike = float(np.mean(np.log(fvar) + rv_proxy / fvar))
    # Benchmark: constant historical volatility from the training sample
    train_sd = float(r_pct.iloc[:-test_size].std())
    qlike_const = float(np.mean(np.log(train_sd ** 2) + rv_proxy / train_sd ** 2))

    # Next-day forecast from the model refitted state through the last observation
    last_var = float(np.asarray(fit.conditional_volatility)[-1] ** 2)
    om, al, be = (float(fit.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    var_t, resid_t = last_var, float(np.asarray(fit.resid)[-1])
    for ret in test.to_numpy():
        var_t = max(om + al * resid_t ** 2 + be * var_t, 1e-12)
        resid_t = float(ret - mu)
    next_var = om + al * resid_t ** 2 + be * var_t
    next_sigma = float(np.sqrt(next_var))
    ann = next_sigma * np.sqrt(252)
    hist_ann = (r_pct.rolling(20).std() * np.sqrt(252)).dropna()
    pct_rank = float((hist_ann < ann).mean() * 100)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in res["dates"]],
        "actual_return_pct": test.tolist(), "garch_sigma_pct": sigma.tolist(),
        "var95_pct": var95.tolist(), "var99_pct": var99.tolist(),
        "realized_vol_20d": test.rolling(20).std().bfill().tolist(),
        "params": {k: float(v) for k, v in fit.params.items()},
        "persistence": float(al + be),
        "backtest": {
            "var95_breach_rate": float(np.mean(loss > var95) * 100),
            "var99_breach_rate": float(np.mean(loss > var99) * 100),
            "var95_breaches": int(np.sum(loss > var95)), "var99_breaches": int(np.sum(loss > var99)),
            "n": int(len(loss)), "qlike_garch": qlike, "qlike_constant_vol": qlike_const,
        },
        "next_day": {
            "sigma_pct": next_sigma, "annualised_pct": float(ann), "percentile_vs_history": pct_rank,
            "var95_pct": float(-(mu + next_sigma * q95)), "var99_pct": float(-(mu + next_sigma * q99)),
            "position_multiplier": float(min(2.0, target_vol_ann / ann)),
            "target_vol_ann": target_vol_ann,
        },
        "hist_ann_vol": {"dates": [d.strftime("%Y-%m-%d") for d in hist_ann.index], "values": hist_ann.tolist()},
    }


# -------------------------------------------------------------- anomalies
def detect_anomalies(returns: pd.Series, window: int = 60, threshold: float = 4.0) -> pd.DataFrame:
    """Rolling robust z-score using the median/MAD of the *previous* `window` returns."""
    prior = returns.shift(1)
    med = prior.rolling(window, min_periods=window).median()
    mad = prior.rolling(window, min_periods=window).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)
    z = 0.6745 * (returns - med) / mad.replace(0, np.nan)
    out = pd.DataFrame({"return_pct": 100 * returns, "robust_z": z})
    out["flag"] = out["robust_z"].abs() >= threshold
    return out
