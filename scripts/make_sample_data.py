# Refresh the offline price snapshot and optionally rebuild the sample portfolio.
#
#   python scripts/make_sample_data.py              # snapshot only
#   python scripts/make_sample_data.py --portfolio  # also rewrite data/portfolio.csv
#
# Sample entry prices are the actual close on each trade date nudged by a few pips of
# made up "execution", so the P&L in the demo comes from real market moves.

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fxdash.market import SNAPSHOT_PATH, fetch_yahoo_rates, pair_price  # noqa: E402

TRADES = [
    # id, date, pair, side, notional (base ccy units), strategy
    ("T001", "2026-01-15", "EURUSD", "BUY", 25_000_000, "G10 Macro"),
    ("T002", "2026-02-03", "USDJPY", "SELL", 20_000_000, "G10 Macro"),
    ("T003", "2026-03-10", "GBPUSD", "BUY", 15_000_000, "G10 Macro"),
    ("T004", "2026-04-07", "AUDUSD", "BUY", 20_000_000, "Commodity FX"),
    ("T005", "2026-05-12", "USDCAD", "SELL", 15_000_000, "Commodity FX"),
    ("T006", "2026-06-02", "EURGBP", "SELL", 10_000_000, "Relative Value"),
    ("T007", "2026-06-23", "USDCHF", "BUY", 10_000_000, "G10 Macro"),
    ("T008", "2026-07-14", "NZDUSD", "SELL", 12_000_000, "Relative Value"),
    ("T009", "2026-08-04", "USDMXN", "SELL", 8_000_000, "EM Carry"),
    ("T010", "2026-09-01", "EURUSD", "SELL", 10_000_000, "Hedge"),
    ("T011", "2026-09-22", "AUDJPY", "BUY", 10_000_000, "Relative Value"),
    ("T012", "2026-10-13", "USDSEK", "BUY", 5_000_000, "G10 Macro"),
    ("T013", "2026-10-16", "USDZAR", "SELL", 6_000_000, "EM Carry"),
]


def main(make_portfolio: bool) -> None:
    rates, mark_time = fetch_yahoo_rates(date.today() - timedelta(days=365 * 3 + 10))
    if mark_time is not None:
        # don't freeze an intraday price into the snapshot, keep completed closes only
        rates = rates.iloc[:-1]
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    rates.round(8).to_csv(SNAPSHOT_PATH)
    print(f"snapshot: {len(rates)} rows to {rates.index[-1]:%Y-%m-%d} -> {SNAPSHOT_PATH}")

    if not make_portfolio:
        return
    rows = []
    for i, (tid, d, pair, side, notional, strat) in enumerate(TRADES):
        px_date = rates.index[rates.index.searchsorted(pd.Timestamp(d))]
        px = float(pair_price(pair, rates.loc[px_date]))
        slip = (1 + 0.0003 * ((i % 3) - 1)) * (1.0001 if side == "BUY" else 0.9999)
        decimals = 3 if pair.endswith("JPY") else 5
        rows.append(dict(trade_id=tid, trade_date=d, ccy_pair=pair, side=side,
                         notional=notional, entry_price=round(px * slip, decimals),
                         strategy=strat))
    out = ROOT / "data" / "portfolio.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"portfolio: {len(rows)} trades -> {out}")


if __name__ == "__main__":
    main("--portfolio" in sys.argv)
