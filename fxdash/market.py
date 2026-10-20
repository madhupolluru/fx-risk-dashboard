# Market data: daily FX history from Yahoo Finance plus a live mark, with an offline
# snapshot to fall back on.
#
# Everything downstream works off one table, `rates`: one column per currency holding the
# USD value of one unit of that currency (EUR -> EURUSD, JPY -> 1/USDJPY, USD -> 1).
# Any pair is then rates[base] / rates[quote], so crosses like EURGBP are priced
# consistently with the USD legs used for risk.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "usd_rates_snapshot.csv"

# currency -> (yahoo ticker, True if quoted as USDxxx and needs inverting)
YAHOO_TICKERS = {
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "AUD": ("AUDUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "JPY": ("USDJPY=X", True),
    "CHF": ("USDCHF=X", True),
    "CAD": ("USDCAD=X", True),
    "SEK": ("USDSEK=X", True),
    "NOK": ("USDNOK=X", True),
    "MXN": ("USDMXN=X", True),
    "ZAR": ("USDZAR=X", True),
    "PLN": ("USDPLN=X", True),
    "SGD": ("USDSGD=X", True),
}
SUPPORTED_CCYS = ["USD", *YAHOO_TICKERS]
LONDON = "Europe/London"


def usd_pair(ccy: str) -> str:
    # market convention for the USD pair of a currency, e.g. EURUSD but USDJPY
    _, inverted = YAHOO_TICKERS[ccy]
    return f"USD{ccy}" if inverted else f"{ccy}USD"


def pair_price(pair: str, rates):
    # quote units per base unit, works on a row (Series) or the whole table
    return rates[pair[:3]] / rates[pair[3:]]


def clean_spikes(rates: pd.DataFrame, threshold: float = 0.08) -> pd.DataFrame:
    # Yahoo FX has the odd bad print. Treat a point as bad if it's more than `threshold`
    # (in log terms) away from the centred 5 day median and replace it with the last good
    # value. The last two rows are left alone because their window is one-sided, so a
    # genuine big move on the latest day would look like a spike.
    med = rates.rolling(5, center=True, min_periods=3).median()
    bad = np.log(rates / med).abs() > threshold
    bad.iloc[-2:] = False
    return rates.mask(bad).ffill()


def relabel_daily_bars(opens: pd.DataFrame) -> pd.DataFrame:
    # Yahoo's daily FX bar for day D is stamped 00:00 London, and for a finished day its
    # "Close" is basically the same as its Open, i.e. the price at midnight London at the
    # *start* of D. Taken at face value every move lands a day late (the Nov 2024 election
    # selloff in EURUSD shows up on 7 Nov) and daily P&L ends up spanning ~1.5 days.
    # So I use the Open of D as the close of the previous business day. Midnight London
    # is 7pm New York, close enough to the usual 5pm NY cut for daily risk.
    out = opens.copy()
    out.index = out.index - pd.offsets.BDay(1)
    return out[~out.index.duplicated(keep="last")]


def _to_usd_values(prices: pd.DataFrame, ccys: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for c in ccys:
        ticker, inverted = YAHOO_TICKERS[c]
        if ticker not in prices or prices[ticker].notna().sum() == 0:
            raise RuntimeError(f"no data for {ticker}")
        s = prices[ticker].astype(float)
        out[c] = 1.0 / s if inverted else s
    return out


def _live_marks(tickers: list[str], ccys: list[str]):
    # latest hourly price for each currency, used as today's mark
    import yfinance as yf

    raw = yf.download(tickers, period="5d", interval="1h", progress=False,
                      auto_adjust=False, threads=True)
    if raw is None or raw.empty:
        return None, None
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.ffill().dropna(how="all")
    if close.empty:
        return None, None
    last = _to_usd_values(close.iloc[[-1]], ccys).iloc[0]
    if last.isna().any():
        return None, None
    stamp = pd.Timestamp(close.index[-1])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return last, stamp


def fetch_yahoo_rates(start: date, ccys: list[str] | None = None):
    # returns (rates, time of the latest mark in UTC or None)
    import yfinance as yf

    ccys = [c for c in (ccys or YAHOO_TICKERS) if c != "USD"]
    tickers = [YAHOO_TICKERS[c][0] for c in ccys]
    raw = yf.download(tickers, start=start.isoformat(), progress=False,
                      auto_adjust=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no daily data")
    opens = raw["Open"]
    if isinstance(opens, pd.Series):
        opens = opens.to_frame(tickers[0])
    opens.index = pd.to_datetime(opens.index).tz_localize(None).normalize()

    daily = _to_usd_values(opens, ccys)
    daily = relabel_daily_bars(daily.sort_index())
    daily = daily[daily.index.dayofweek < 5].dropna(how="all").ffill().dropna()
    if len(daily) < 300:
        raise RuntimeError(f"only {len(daily)} days of history came back")

    # today's mark comes from the hourly bars. If that fails the last close is used.
    mark_time = None
    try:
        live, stamp = _live_marks(tickers, ccys)
    except Exception:
        live, stamp = None, None
    if live is not None:
        day = pd.Timestamp(stamp.tz_convert(LONDON).date())
        day = pd.offsets.BDay().rollforward(day)  # Sunday evening trades for Monday
        if day >= daily.index[-1]:
            daily.loc[day] = live
            mark_time = stamp

    daily = clean_spikes(daily.sort_index())
    daily.insert(0, "USD", 1.0)
    daily.index.name = "date"
    return daily, mark_time


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


@dataclass
class MarketData:
    rates: pd.DataFrame
    source: str
    live: bool
    mark_time: pd.Timestamp | None = None  # UTC time of the latest price, if intraday
    notes: list[str] = field(default_factory=list)

    @property
    def asof(self) -> pd.Timestamp:
        return self.rates.index[-1]


def get_market_data(years: int = 3, prefer_live: bool = True) -> MarketData:
    start = date.today() - timedelta(days=365 * years + 10)
    notes = []
    if prefer_live:
        try:
            rates, mark_time = fetch_yahoo_rates(start)
            md = MarketData(rates, "Yahoo Finance", True, mark_time, notes)
        except Exception as exc:  # no network, rate limited, schema change etc.
            notes.append(f"Live prices unavailable ({type(exc).__name__}: {exc}), "
                         "using the bundled snapshot.")
            md = None
    else:
        md = None
    if md is None:
        snap = load_snapshot()
        md = MarketData(snap, f"bundled snapshot to {snap.index[-1]:%d %b %Y}", False, None,
                        notes)

    age = (pd.Timestamp(date.today()) - md.asof).days
    if age > 4:
        notes.append(f"Latest price is {age} days old ({md.asof:%d %b %Y}).")
    return md
