# Portfolio loading, valuation and P&L for FX spot positions.
#
# A trade of N units of BASE at entry price K (QUOTE per BASE) is long N BASE and short
# N*K QUOTE. With b = USD per BASE and q = USD per QUOTE its USD value is
#
#     V = N*b - N*K*q = N*q*(S - K)     where S = b/q is the pair price
#
# V is linear in b and q, which gives a few things for free:
#   inception P&L = V today (quote ccy P&L translated at today's rate)
#   daily P&L     = V today - V at the previous close (0 before the trade existed)
#   exposures     = {BASE: N*b, QUOTE: -N*K*q}. A % move r in each currency's USD value
#                   changes V by exactly sum(exposure * r), so the VaR built on these is a
#                   full revaluation for spot, not a delta approximation.

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .market import SUPPORTED_CCYS, pair_price

REQUIRED_COLS = ["trade_id", "trade_date", "ccy_pair", "side", "notional", "entry_price"]
SIDE_ALIASES = {"BUY": "BUY", "B": "BUY", "LONG": "BUY", "SELL": "SELL", "S": "SELL",
                "SHORT": "SELL"}


class PortfolioError(ValueError):
    pass


# one-line trade tickets for the what-if box, e.g. "BUY 25m EURUSD", "sell 5,000,000 usd/jpy",
# "buy 10m EURGBP @ 0.86". Anything after a "·" is ignored, so the hedge labels parse too.
TICKET = re.compile(r"^\s*(BUY|SELL|B|S|LONG|SHORT)\s+([\d.,]+)\s*(K|M|MM|BN|B)?\s+"
                    r"([A-Z]{3})[/\- ]?([A-Z]{3})(?:\s*@\s*([\d.]+))?\s*$", re.IGNORECASE)
UNITS = {"": 1.0, "K": 1e3, "M": 1e6, "MM": 1e6, "B": 1e9, "BN": 1e9}


def parse_ticket(text: str) -> dict:
    body = str(text).split("·")[0].strip()
    m = TICKET.match(body)
    if not m:
        raise PortfolioError(f'Couldn\'t read "{body}". Try something like BUY 10m EURJPY.')
    side = SIDE_ALIASES[m[1].upper()]
    notional = float(m[2].replace(",", "")) * UNITS[(m[3] or "").upper()]
    pair = (m[4] + m[5]).upper()
    if pair[:3] not in SUPPORTED_CCYS or pair[3:] not in SUPPORTED_CCYS or pair[:3] == pair[3:]:
        raise PortfolioError(f"{pair} isn't a supported pair.")
    if not notional > 0:
        raise PortfolioError("Notional must be more than zero.")
    price = float(m[6]) if m[6] else None
    return {"side": side, "notional": notional, "ccy_pair": pair, "entry_price": price}


def _read(src) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pd.read_csv(src)


def _to_number(s: pd.Series) -> pd.Series:
    # copes with "25,000,000" or "25000000 " from a spreadsheet export
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "").str.strip()
    return pd.to_numeric(s, errors="coerce")


def load_portfolio(src, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    # read and validate a trade file (path, buffer, bytes or DataFrame).
    # raises PortfolioError listing every problem found, not just the first one.
    try:
        df = _read(src)
    except Exception as exc:
        raise PortfolioError(f"could not read the file ({exc})") from exc

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise PortfolioError(f"missing columns: {', '.join(missing)}")

    # drop rows that are completely empty (blank lines, rows deleted in the editor)
    df = df[~df[REQUIRED_COLS[1:]].isna().all(axis=1)].copy()
    if df.empty:
        raise PortfolioError("no trades in the file")

    if "strategy" not in df:
        df["strategy"] = "Unassigned"
    df["strategy"] = df["strategy"].fillna("Unassigned").astype(str).str.strip()
    df.loc[df["strategy"] == "", "strategy"] = "Unassigned"

    # blank ids get a generated one so a half filled what-if row still works
    ids = df["trade_id"].astype("string").str.strip()
    blank = ids.isna() | (ids == "")
    ids[blank] = [f"NEW{i + 1}" for i in range(blank.sum())]
    df["trade_id"] = ids.astype(str)

    df["ccy_pair"] = (df["ccy_pair"].astype(str).str.upper()
                      .str.replace(r"[^A-Z]", "", regex=True))
    df["side"] = df["side"].astype(str).str.upper().str.strip().map(SIDE_ALIASES)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["notional"] = _to_number(df["notional"])
    df["entry_price"] = _to_number(df["entry_price"])

    problems = []
    for _, r in df.iterrows():
        tid, pair = r["trade_id"], r["ccy_pair"]
        if (len(pair) != 6 or pair[:3] not in SUPPORTED_CCYS or pair[3:] not in SUPPORTED_CCYS
                or pair[:3] == pair[3:]):
            problems.append(f"{tid}: unsupported pair '{pair}'")
        if pd.isna(r["side"]):
            problems.append(f"{tid}: side must be BUY or SELL")
        if not r["notional"] > 0:
            problems.append(f"{tid}: notional must be a positive number of base ccy units")
        if not r["entry_price"] > 0:
            problems.append(f"{tid}: entry_price must be positive")
        if pd.isna(r["trade_date"]):
            problems.append(f"{tid}: invalid trade_date")
        elif asof is not None and r["trade_date"] > asof:
            problems.append(f"{tid}: trade_date {r['trade_date']:%Y-%m-%d} is after the "
                            f"latest market date {asof:%Y-%m-%d}")
    dupes = df.loc[df["trade_id"].duplicated(), "trade_id"].unique()
    if len(dupes):
        problems.append(f"duplicate trade_id: {', '.join(dupes)}")
    if problems:
        raise PortfolioError("; ".join(problems))

    df["base"] = df["ccy_pair"].str[:3]
    df["quote"] = df["ccy_pair"].str[3:]
    df["signed_notional"] = np.where(df["side"] == "BUY", 1.0, -1.0) * df["notional"]
    return df.set_index("trade_id")


def fill_at_market(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    # for what-if rows: a missing trade date means today and a missing entry price means
    # the market close on the trade date, so a new trade starts at (close to) zero P&L
    df = df.copy()
    today = rates.index[-1]
    dates = pd.to_datetime(df["trade_date"], errors="coerce")
    df["trade_date"] = dates.fillna(today)
    px = pd.to_numeric(df["entry_price"], errors="coerce")
    for i in df.index[px.isna()]:
        pair = str(df.at[i, "ccy_pair"]).upper().replace("/", "")
        if len(pair) != 6 or pair[:3] not in rates or pair[3:] not in rates:
            continue
        row = min(rates.index.searchsorted(df.at[i, "trade_date"]), len(rates) - 1)
        px[i] = float(pair_price(pair, rates.iloc[row]))
    df["entry_price"] = px
    return df


def sanity_warnings(pf: pd.DataFrame, rates: pd.DataFrame, tol: float = 0.05) -> list[str]:
    # things that won't break the maths but a PM would want flagged
    out = []
    first = rates.index[0]
    for tid, p in pf.iterrows():
        if p["trade_date"] < first:
            out.append(f"{tid} predates the price history ({first:%d %b %Y}); inception P&L "
                       "is fine but the P&L history chart starts late.")
            continue
        row = min(rates.index.searchsorted(p["trade_date"]), len(rates) - 1)
        mkt = float(pair_price(p["ccy_pair"], rates.iloc[row]))
        if abs(p["entry_price"] / mkt - 1) > tol:
            out.append(f"{tid} entry {p['entry_price']:.6g} is {p['entry_price'] / mkt - 1:+.0%} "
                       f"from the {p['ccy_pair']} close on the trade date ({mkt:.6g}). "
                       "Typo or inverted quote?")
    return out


def value_history(pf: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    # USD value of each trade on every date, zero before its trade date.
    # rows are dates, columns trade ids. The row sum is the book's cumulative P&L.
    out = pd.DataFrame(0.0, index=rates.index, columns=pf.index)
    for tid, p in pf.iterrows():
        v = p["signed_notional"] * (rates[p["base"]] - p["entry_price"] * rates[p["quote"]])
        start = int(rates.index.searchsorted(p["trade_date"]))
        v.iloc[:start] = 0.0
        out[tid] = v
    return out


def value_positions(pf: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    # current marks and P&L for each trade, using the last two rows of rates
    if len(rates) < 2:
        raise ValueError("need at least two days of rates")
    today, yday = rates.iloc[-1], rates.iloc[-2]
    hist = value_history(pf, rates.iloc[-2:])
    out = pf.copy()
    out["spot"] = [float(pair_price(p, today)) for p in pf["ccy_pair"]]
    out["prev_close"] = [float(pair_price(p, yday)) for p in pf["ccy_pair"]]
    out["pnl_inception_usd"] = hist.iloc[-1]
    # for a trade booked since the last close this is the same as inception P&L
    out["pnl_daily_usd"] = hist.iloc[-1] - hist.iloc[-2]
    out["pnl_quote"] = pf["signed_notional"] * (out["spot"] - pf["entry_price"])
    out["usd_notional"] = pf["signed_notional"] * today[pf["base"]].to_numpy()
    out["move_pct"] = (out["spot"] / pf["entry_price"] - 1.0) * np.sign(pf["signed_notional"])
    out["days_held"] = (rates.index[-1] - pf["trade_date"]).dt.days.clip(lower=0)
    return out


def usd_exposures(pf: pd.DataFrame, rates_row: pd.Series) -> pd.DataFrame:
    # USD exposure of each trade to each non-USD currency (rows are trade ids).
    # only currencies with some exposure are kept, so a flat book gives no columns.
    ccys = [c for c in rates_row.index if c != "USD"]
    exp = pd.DataFrame(0.0, index=pf.index, columns=ccys)
    for tid, p in pf.iterrows():
        n = p["signed_notional"]
        if p["base"] != "USD":
            exp.at[tid, p["base"]] += n * rates_row[p["base"]]
        if p["quote"] != "USD":
            exp.at[tid, p["quote"]] -= n * p["entry_price"] * rates_row[p["quote"]]
    used = exp.columns[(exp.abs() > 1e-6).any(axis=0)]
    return exp[used]


def daily_pnl_attribution(pf: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
    # Splits today's P&L by currency. For trades that existed at the previous close the
    # split is exact: dV = sum over ccys of (exposure at yesterday's rates) * (% move).
    # Trades booked since then have no previous value, their P&L goes in "New trades".
    yday_date = rates.index[-2]
    old = pf[pf["trade_date"] <= yday_date]
    new = pf[pf["trade_date"] > yday_date]
    moves = rates.iloc[-1] / rates.iloc[-2] - 1
    out = pd.Series(dtype=float)
    if len(old):
        exp = usd_exposures(old, rates.iloc[-2])
        out = (exp.sum(axis=0) * moves[exp.columns]).sort_values()
    if len(new):
        out["New trades"] = value_history(new, rates.iloc[-1:]).iloc[-1].sum()
    return out


def net_by_pair(pos: pd.DataFrame) -> pd.DataFrame:
    # aggregate trades in the same pair. breakeven is the spot where the net position's
    # P&L is zero: sum(n_i * (S - K_i)) = 0  ->  S = sum(n_i * K_i) / sum(n_i)
    g = pos.assign(nk=pos["signed_notional"] * pos["entry_price"]).groupby("ccy_pair")
    out = pd.DataFrame({
        "trades": g.size(),
        "net_notional": g["signed_notional"].sum(),
        "nk": g["nk"].sum(),
        "spot": g["spot"].last(),
        "usd_notional": g["usd_notional"].sum(),
        "pnl_daily_usd": g["pnl_daily_usd"].sum(),
        "pnl_inception_usd": g["pnl_inception_usd"].sum(),
    })
    flat = out["net_notional"].abs() < 1e-6
    out["breakeven"] = np.where(flat, np.nan, out["nk"] / out["net_notional"].where(~flat, 1))
    return out.drop(columns="nk")
