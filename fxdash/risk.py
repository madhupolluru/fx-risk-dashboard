# VaR engine.
#
# Inputs are the USD exposure matrix E (rows = positions or any other grouping, columns =
# currencies, see portfolio.usd_exposures) and daily % changes R of each currency's USD
# value. P&L of row i under day h is sum_c E[i, c] * R[h, c], which for spot FX is an
# exact revaluation.
#
# Three methods:
#   hist  - plain historical simulation over the last `lookback` days
#   fhs   - filtered historical simulation (Hull & White 1998): each past day's moves are
#           rescaled by (EWMA vol now / EWMA vol on that day), so the scenarios keep the
#           real fat tails and co-movements but reflect today's volatility level
#   param - variance-covariance with a RiskMetrics EWMA covariance (lambda 0.94)
#
# VaR and ES are positive USD losses. Multi-day figures use sqrt(h) scaling.
# Component VaR is an Euler allocation in every method, so components add up to VaR.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .market import usd_pair

EWMA_LAMBDA = 0.94
METHODS = {
    "hist": "Historical simulation",
    "fhs": "Filtered historical (vol-scaled)",
    "param": "Parametric (EWMA)",
}


def method_key(method: str) -> str:
    m = method.lower()
    if m.startswith("param"):
        return "param"
    if m.startswith("fhs") or m.startswith("filtered"):
        return "fhs"
    return "hist"


def daily_returns(rates: pd.DataFrame, ccys) -> pd.DataFrame:
    # full history of daily % moves, the methods pick their own window from it
    return rates[list(ccys)].pct_change().dropna()


def ewma_cov(returns: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    X = returns.to_numpy()
    w = lam ** np.arange(len(X) - 1, -1, -1)
    w /= w.sum()
    cov = (X * w[:, None]).T @ X  # zero mean, the usual assumption for daily FX
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_vol(returns: pd.DataFrame, lam: float = EWMA_LAMBDA, seed: int = 20):
    # One-step-ahead EWMA vol per currency. Row t is the forecast for day t made with data
    # up to t-1, so it can be used to standardise day t without look-ahead.
    # Also returns the forecast for the next (unseen) day.
    X = returns.to_numpy()
    var = np.empty_like(X)
    v = np.mean(X[:seed] ** 2, axis=0)
    for t in range(len(X)):
        var[t] = v
        v = lam * v + (1 - lam) * X[t] ** 2
    sig = pd.DataFrame(np.sqrt(var), index=returns.index, columns=returns.columns)
    return sig, pd.Series(np.sqrt(v), index=returns.columns)


def fhs_scenarios(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    sig, sig_next = ewma_vol(returns)
    z = returns / sig
    return z.iloc[-lookback:] * sig_next


def scenarios_for(method: str, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if method_key(method) == "fhs":
        return fhs_scenarios(returns, lookback)
    return returns.iloc[-lookback:]


def hist_var(pnl, alpha: float) -> float:
    return float(-np.quantile(pnl, 1 - alpha))


def hist_es(pnl, alpha: float) -> float:
    pnl = np.asarray(pnl)
    return float(-pnl[pnl <= np.quantile(pnl, 1 - alpha)].mean())


def scenario_pnl(exposures: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    # P&L of each row of `exposures` under each scenario day (days x rows)
    R = scen[exposures.columns].to_numpy()
    return pd.DataFrame(R @ exposures.to_numpy().T, index=scen.index, columns=exposures.index)


@dataclass
class RiskResult:
    method: str
    alpha: float
    horizon: int
    var: float
    es: float
    positions: pd.DataFrame   # standalone / component / incremental / marginal per row
    currencies: pd.DataFrame  # exposure / standalone / component per currency
    scenarios: pd.DataFrame   # the 1-day scenario returns used (empty for parametric)

    @property
    def diversification(self) -> float:
        return float(self.positions["standalone_var"].sum() - self.var)

    def portfolio_pnl(self, exposures: pd.DataFrame) -> pd.Series:
        # 1-day scenario P&L of the whole book
        return scenario_pnl(exposures, self.scenarios).sum(axis=1)


def _component_hist(pnl: pd.DataFrame, alpha: float, k: int | None = None):
    # Component VaR for historical simulation. The textbook version takes each position's
    # P&L on the single VaR day, which is very noisy. Here I average over the few scenarios
    # either side of the VaR quantile and rescale so the components sum to VaR.
    port = pnl.sum(axis=1).to_numpy()
    var = hist_var(port, alpha)
    T = len(port)
    k = k if k is not None else max(1, T // 200)
    order = np.argsort(port)
    centre = int(np.clip(round((1 - alpha) * (T - 1)), 0, T - 1))
    idx = order[max(0, centre - k): centre + k + 1]
    raw = -pnl.iloc[idx].mean(axis=0)
    comp = raw * (var / raw.sum()) if abs(raw.sum()) > 1e-12 else raw
    return var, comp


def _historical(exposures, scen, alpha):
    pnl = scenario_pnl(exposures, scen)
    port = pnl.sum(axis=1)
    var, comp = _component_hist(pnl, alpha)
    es = hist_es(port.to_numpy(), alpha)

    pos = pd.DataFrame(index=exposures.index)
    pos["standalone_var"] = [hist_var(pnl[c], alpha) for c in pnl.columns]
    pos["component_var"] = comp
    pos["incremental_var"] = [var - hist_var(port - pnl[c], alpha) for c in pnl.columns]
    pos["component_es"] = -pnl[port <= np.quantile(port, 1 - alpha)].mean(axis=0)

    x = exposures.sum(axis=0)
    ccy_pnl = scen[exposures.columns] * x
    _, ccy_comp = _component_hist(ccy_pnl, alpha)
    ccy = pd.DataFrame({
        "exposure_usd": x,
        "standalone_var": [hist_var(ccy_pnl[c], alpha) for c in ccy_pnl.columns],
        "component_var": ccy_comp,
    })
    return var, es, pos, ccy


def _parametric(exposures, returns, alpha):
    z = stats.norm.ppf(alpha)
    es_mult = stats.norm.pdf(z) / (1 - alpha)  # ES / sigma for a normal
    cov = ewma_cov(returns[exposures.columns]).to_numpy()
    E = exposures.to_numpy()
    x = E.sum(axis=0)
    sigma = float(np.sqrt(max(x @ cov @ x, 0.0)))
    var = z * sigma
    beta = cov @ x / sigma if sigma > 0 else np.zeros_like(x)  # d sigma / d x

    pos = pd.DataFrame(index=exposures.index)
    pos["standalone_var"] = z * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", E, cov, E), 0))
    pos["component_var"] = z * (E @ beta)
    pos["incremental_var"] = [var - z * np.sqrt(max((x - e) @ cov @ (x - e), 0.0)) for e in E]
    pos["component_es"] = pos["component_var"] * es_mult / z

    ccy = pd.DataFrame({
        "exposure_usd": x,
        "standalone_var": z * np.abs(x) * np.sqrt(np.diag(cov)),
        "component_var": z * x * beta,
    }, index=exposures.columns)
    return var, sigma * es_mult, pos, ccy


def compute_risk(method: str, exposures: pd.DataFrame, returns: pd.DataFrame, alpha: float,
                 horizon: int, lookback: int, notionals: pd.Series | None = None) -> RiskResult:
    key = method_key(method)
    cols = ["standalone_var", "component_var", "incremental_var", "component_es"]

    if exposures.shape[1] == 0:  # flat book, nothing to measure
        pos = pd.DataFrame(0.0, index=exposures.index, columns=cols)
        ccy = pd.DataFrame(columns=["exposure_usd", "standalone_var", "component_var"])
        var, es, scen = 0.0, 0.0, pd.DataFrame()
    elif key == "param":
        scen = pd.DataFrame()
        var, es, pos, ccy = _parametric(exposures, returns.iloc[-lookback:], alpha)
    else:
        scen = scenarios_for(key, returns[exposures.columns], lookback)
        var, es, pos, ccy = _historical(exposures, scen, alpha)

    scale = np.sqrt(horizon)
    pos = pos.copy()
    ccy = ccy.copy()
    pos[cols] = pos[cols] * scale
    ccy[["standalone_var", "component_var"]] = ccy[["standalone_var", "component_var"]] * scale
    var, es = var * scale, es * scale
    pos["pct_of_var"] = pos["component_var"] / var if var else 0.0
    ccy["pct_of_var"] = ccy["component_var"] / var if var else 0.0
    if notionals is not None:
        # Euler: component_i = n_i * dVaR/dn_i, so dVaR/dn_i = component_i / n_i.
        # quoted as the VaR change from buying another 1m of the base currency.
        pos["marginal_var_per_1m"] = pos["component_var"] / notionals.reindex(pos.index) * 1e6
    return RiskResult(METHODS[key], alpha, horizon, var, es, pos, ccy, scen)


def var_of(x: pd.Series, method: str, returns: pd.DataFrame, alpha: float, horizon: int,
           lookback: int) -> float:
    # VaR of a single net currency exposure vector
    x = x[x.abs() > 1e-6]
    if x.empty:
        return 0.0
    return compute_risk(method, x.to_frame().T, returns, alpha, horizon, lookback).var


def stressed_var(exposures: pd.DataFrame, returns: pd.DataFrame, alpha: float, horizon: int,
                 window: int = 250):
    # Historical VaR of today's book on the worst `window` days in the history we have.
    # (Basel-style stressed VaR, limited to the ~3 years of data the app downloads.)
    # returns (var, window start, window end)
    if exposures.shape[1] == 0 or len(returns) < window:
        return 0.0, None, None
    port = returns[exposures.columns] @ exposures.sum(axis=0)
    q = -port.rolling(window).quantile(1 - alpha)
    end = q.idxmax()
    start = port.index[port.index.get_loc(end) - window + 1]
    return float(q.max()) * np.sqrt(horizon), start, end


# hedging

def best_hedges(exposures: pd.DataFrame, returns: pd.DataFrame, rates_row: pd.Series,
                method: str, alpha: float, horizon: int, lookback: int,
                min_size: float = 1e5) -> pd.DataFrame:
    # "Best single hedge" report (Litterman, Hot Spots and Hedges, 1996). For each currency
    # c the variance-minimising hedge is a USD exposure h = -cov(c, book) / var(c). Every
    # currency we have prices for is a candidate, not just the ones in the book, so e.g.
    # NZD can show up as a cheap proxy hedge for AUD. The VaR after the hedge is recomputed
    # with the selected method, so it is a proper revaluation and not just the variance
    # approximation used to size the trade.
    ccys = list(returns.columns)
    x = exposures.sum(axis=0).reindex(ccys).fillna(0.0)
    if (x.abs() < 1e-6).all():
        return pd.DataFrame()
    key = method_key(method)
    if key == "param":
        cov = ewma_cov(returns.iloc[-lookback:])
    else:
        S = scenarios_for(key, returns, lookback)
        cov = S.T @ S / len(S)
    var_now = var_of(x, method, returns, alpha, horizon, lookback)

    rows = []
    for c in ccys:
        if cov.at[c, c] <= 0:
            continue
        h = -float(cov.loc[c] @ x) / cov.at[c, c]
        if abs(h) < min_size:
            continue
        x_new = x.copy()
        x_new[c] += h
        var_new = var_of(x_new, method, returns, alpha, horizon, lookback)
        pair, side, notional = hedge_trade(c, h, rates_row)
        rows.append({"ccy": c, "hedge_usd": h, "pair": pair, "side": side,
                     "notional": notional, "var_after": var_new,
                     "var_change": var_new - var_now,
                     "pct_reduction": 1 - var_new / var_now if var_now else 0.0})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("var_after").reset_index(drop=True)


def hedge_trade(ccy: str, usd_exposure: float, rates_row: pd.Series, round_to: float = 1e5):
    # turn a USD exposure to `ccy` into a spot trade in the USD pair.
    # XXXUSD: buying N gives exposure N*b, so N = h/b.
    # USDXXX: buying N USD gives XXX exposure of -N, so sell N = h.
    pair = usd_pair(ccy)
    if pair.startswith(ccy):
        n = usd_exposure / rates_row[ccy]
        side = "BUY" if n > 0 else "SELL"
    else:
        n = usd_exposure
        side = "SELL" if n > 0 else "BUY"
    notional = max(round(abs(n) / round_to) * round_to, round_to)
    return pair, side, notional


# backtesting

def kupiec_pof(exceptions: int, n: int, alpha: float) -> float:
    # p-value of Kupiec's proportion of failures test, H0: exception rate = 1 - alpha
    if n == 0:
        return float("nan")
    p, x = 1 - alpha, exceptions

    def ll(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    lr = -2 * (ll(p) - ll(x / n))
    return float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen(hits) -> tuple[float, float]:
    # Christoffersen (1998) independence test: are exceptions clustered? Compares the
    # chance of an exception after an exception with the chance after a quiet day.
    # returns (LR statistic, p-value). Kupiec only checks the count, this checks timing.
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        return float("nan"), float("nan")
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    if n01 + n11 == 0:
        return 0.0, 1.0  # no exceptions, nothing to cluster

    def ll(p, a, b):  # a "no exception" days, b exceptions
        p = min(max(p, 1e-12), 1 - 1e-12)
        return a * np.log(1 - p) + b * np.log(p)

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def basel_zone(exceptions: int, n: int, alpha: float) -> str | None:
    # Basel traffic light, defined for 250 days at 99%
    if abs(alpha - 0.99) > 1e-9 or n != 250:
        return None
    return "Green" if exceptions <= 4 else "Amber" if exceptions <= 9 else "Red"


def backtest(exposures: pd.DataFrame, returns: pd.DataFrame, method: str, alpha: float,
             lookback: int, n_test: int = 250) -> pd.DataFrame:
    # Out-of-sample 1-day VaR backtest of the *current* book (hypothetical P&L).
    # For each of the last n_test days the VaR uses only data from before that day.
    cols = ["pnl", "var", "exception"]
    if exposures.shape[1] == 0:
        return pd.DataFrame(columns=cols)
    key = method_key(method)
    R = returns[exposures.columns]
    x = exposures.sum(axis=0).to_numpy()
    port = R.to_numpy() @ x
    n_test = min(n_test, len(R) - lookback)
    if n_test <= 0:
        return pd.DataFrame(columns=cols)

    z = stats.norm.ppf(alpha)
    if key == "fhs":
        sig, _ = ewma_vol(R)  # causal, row t only uses data up to t-1
        std = (R / sig).to_numpy()
        sig = sig.to_numpy()
    rows = []
    for t in range(len(R) - n_test, len(R)):
        w = slice(t - lookback, t)
        if key == "param":
            cov = ewma_cov(R.iloc[w]).to_numpy()
            v = z * np.sqrt(x @ cov @ x)
        elif key == "fhs":
            v = hist_var((std[w] * sig[t]) @ x, alpha)
        else:
            v = hist_var(port[w], alpha)
        rows.append((R.index[t], port[t], v))
    bt = pd.DataFrame(rows, columns=["date", "pnl", "var"]).set_index("date")
    bt["exception"] = bt["pnl"] < -bt["var"]
    return bt


# stress tests

G10_EM = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "MXN", "ZAR", "PLN",
          "SGD"]

# moves are % changes in each currency's value against USD (+ = currency strengthens)
STRESS_SCENARIOS = {
    "Broad USD rally (+5%)": {c: -0.05 for c in G10_EM},
    "Broad USD selloff (-5%)": {c: 0.05 for c in G10_EM},
    "Risk-off: havens up, commodity and EM FX down": {
        "JPY": 0.07, "CHF": 0.04, "EUR": -0.01, "GBP": -0.03, "AUD": -0.07, "NZD": -0.07,
        "CAD": -0.03, "SEK": -0.05, "NOK": -0.06, "MXN": -0.12, "ZAR": -0.12, "PLN": -0.06,
        "SGD": -0.02},
    "Dovish ECB surprise (EUR -4%)": {"EUR": -0.04, "CHF": -0.015, "SEK": -0.03, "NOK": -0.025,
                                      "PLN": -0.035, "GBP": -0.01},
    "UK political shock (GBP -8%)": {"GBP": -0.08, "EUR": -0.02},
    "BoJ hawkish surprise (JPY +6%)": {"JPY": 0.06, "AUD": -0.01, "NZD": -0.01},
    "EM selloff (MXN/ZAR -15%)": {"MXN": -0.15, "ZAR": -0.15, "PLN": -0.05, "AUD": -0.02,
                                  "CAD": -0.015, "NOK": -0.02},
    "Oil crash (NOK/CAD -6%)": {"NOK": -0.06, "CAD": -0.04, "MXN": -0.04, "AUD": -0.01},
}

# real episodes inside the price history, replayed close to close on today's book
HISTORICAL_EPISODES = {
    "Yen carry unwind (31 Jul - 5 Aug 2024)": ("2024-07-31", "2024-08-05"),
    "US election (5 - 6 Nov 2024)": ("2024-11-05", "2024-11-06"),
    "Liberation Day tariffs (2 - 11 Apr 2025)": ("2025-04-02", "2025-04-11"),
}


def episode_moves(rates: pd.DataFrame, start: str, end: str) -> pd.Series | None:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s < rates.index[0] or e > rates.index[-1]:
        return None
    a = rates.iloc[rates.index.searchsorted(s)]
    b = rates.iloc[rates.index.searchsorted(e)]
    return (b / a - 1).drop("USD", errors="ignore")


def worst_window(exposures: pd.DataFrame, returns: pd.DataFrame, days: int = 5):
    # worst run of `days` consecutive days for today's book in the whole history
    # returns (pnl, start, end) using compounded moves, so it's an exact revaluation
    if exposures.shape[1] == 0 or len(returns) <= days:
        return 0.0, None, None
    growth = (1 + returns[exposures.columns]).rolling(days).apply(np.prod, raw=True) - 1
    pnl = growth.dropna() @ exposures.sum(axis=0)
    end = pnl.idxmin()
    start = returns.index[returns.index.get_loc(end) - days + 1]
    return float(pnl.min()), start, end


def stress_pnl(exposures: pd.DataFrame, shocks) -> pd.Series:
    s = pd.Series(shocks, dtype=float).reindex(exposures.columns).fillna(0.0)
    return exposures @ s
