# FX Portfolio Risk Dashboard

A Streamlit dashboard for an FX portfolio manager to keep an eye on P&L and risk for a
book of FX spot trades.

Live dashboard: https://madhupolluru-fx-risk-dashboard.streamlit.app

## Running it

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

Tests (P&L maths, VaR, backtests, loader): `pip install pytest && pytest -q`

`python scripts/make_sample_data.py` refreshes the offline price snapshot, and with
`--portfolio` it also rebuilds the sample book.

## What's in it

The header shows inception and daily P&L, gross notional, VaR, expected shortfall,
stressed VaR and how much of the VaR limit is used. It stays pinned to the top together
with the tab bar, so the numbers and tabs are always in view while you scroll. There are
eight tabs.

- Overview: a list of things that need attention (limit usage, model failing its
  backtest, vol picking up in a currency you hold, concentration, unusually large moves,
  suspicious trade data), today's P&L split by currency, cumulative P&L and the biggest
  risk contributors.
- Positions: trade blotter with entry vs spot, P&L in quote currency and USD, and each
  trade's share of VaR. Net position by pair with a breakeven level, net currency
  exposure, and a strategy table with strategy-level VaR and P&L per unit of risk.
- Risk: standalone vs component VaR by trade and by currency, the full decomposition
  (component, marginal, incremental, component ES), the P&L distribution against a normal
  curve, all three VaR models side by side, stressed VaR, correlations and a
  realised vs EWMA vol table.
- Hedging: the best single hedge in every currency (including ones not in the book) and
  what it does to VaR. Under "Test a trade" you can pick one of those hedges, type a
  trade straight into the box (`BUY 10m EURJPY`, `SELL 5m USDJPY @ 150`) or fill in a
  small form, see what it does to VaR and P&L first, and then add it to the what-if
  book. Trades you've added are listed with a Remove button, and a trade that's already
  in the book can't be added twice. Also what closing each trade does to VaR, and
  marginal VaR.
- Stress: hypothetical macro scenarios, replays of real episodes from the price history
  (Aug 2024 yen carry unwind, Nov 2024 US election, Apr 2025 tariffs), the worst 5-day run
  for today's book, a custom shock builder and the 10 worst days in the lookback.
- P&L & backtest: cumulative P&L by strategy, with the whole book shaded green while
  it's up and red while it's down. Best and worst day, hit rate, P&L Sharpe, drawdown.
  An out-of-sample VaR backtest with Kupiec, Christoffersen and the Basel traffic light.
- Market: 1D/1W/1M/YTD moves, today's move in vol units, realised vols, position in the
  1-year range, and a price chart with entries and the net breakeven marked.
- What-if: an editable copy of the book. Add, amend or delete trades and everything
  updates, with the change vs the original book shown under each number in the header.
  Leaving entry price blank books at the market, so it works as a pre-trade check. An
  Exit what-if button in the header goes back to the original book from any tab.

The sidebar has the VaR method, confidence (90 to 99%), horizon (1 to 20 days), lookback
(100 to 750 trading days), the VaR limit and a live vs snapshot data switch. You can also
upload your own CSV.

Charts have no toolbar: drag to move along, pinch (or Ctrl + scroll) to zoom, and
double-click to reset. A normal scroll over a chart just scrolls the page.

## Methodology

### Portfolio file

`data/portfolio.csv` has `trade_id, trade_date, ccy_pair, side, notional, entry_price` and
an optional `strategy`. Notional is in base currency units and entry price is quote per
base, the way the pair is quoted. The loader is fairly forgiving (EUR/USD, eur-usd, B/S,
LONG/SHORT, "25,000,000" all work) but strict about anything that would give a wrong
number: unknown pairs, non-positive notionals or prices, bad or future dates and duplicate
ids are all rejected with a list of every problem. It also warns, without rejecting, if an
entry price is more than 5% away from the market on the trade date, which is usually a
typo or an inverted quote.

### Market data

Daily prices for 13 currencies against USD (EUR, GBP, JPY, CHF, AUD, NZD, CAD, SEK, NOK,
MXN, ZAR, PLN, SGD) come from Yahoo Finance via yfinance. Everything is stored as USD per
unit of currency, and any pair, crosses included, is just rate[base] / rate[quote]. That
way marks and risk always use the same numbers.

One thing worth knowing about Yahoo FX data: the daily bar for a date D is stamped at
00:00 London and for a finished day its "Close" is really the price at the start of D
(it's the same as the Open). Taken at face value, every move shows up a day late. The Nov
2024 election selloff in EURUSD appears on 7 Nov, for example, and "daily P&L" ends up
covering about a day and a half. I use each bar's Open as the close of the previous
business day instead. That's midnight London / 7pm New York, close enough to the usual
5pm NY cut. Today's mark comes from the latest hourly bar, so during the day the daily
P&L is live vs last night's close. There's a test for this in `tests/test_calcs.py`.

Isolated bad ticks are removed with a rolling median filter. Prices are cached for 10
minutes. If Yahoo fails or a ticker is missing, the app falls back to a 3-year snapshot in
`data/` and says so in the alerts.

### P&L

For a trade of N base units at entry price K, with b and q the USD value of one unit of
the base and quote currency:

    V = N*b - N*K*q = N*(S - K)*q      where S = b/q

- Inception P&L is V today, i.e. the quote currency P&L converted at today's rate.
- Daily P&L is V today minus V at the previous close. For a trade booked since the last
  close it's the same as inception P&L.
- The daily split by currency is exact, not an approximation: because V is linear in b
  and q, the change in V is exactly the sum of each currency's exposure at yesterday's
  rates times its % move. Trades booked since the last close are shown separately.

### Risk factors and exposures

The same linearity means each trade breaks down exactly into USD exposures,
base: N*b and quote: -N*K*q (USD legs dropped). The risk factors are the daily % changes
in each currency's USD value. Crosses net automatically, e.g. a EURGBP trade shares its
EUR exposure with EURUSD trades. For spot, exposure times return is a full revaluation,
not a delta approximation.

### VaR

Three methods, chosen in the sidebar:

- Historical simulation (default): apply each of the last N days of moves to today's
  exposures. VaR is the (1 - confidence) quantile of the losses and ES is the average
  loss beyond it. No distribution is assumed, so fat tails and real co-movements are
  kept, but it is slow to react when vol changes.
- Filtered historical simulation (Hull and White, 1998): same, but each past day's moves
  are scaled by (EWMA vol today / EWMA vol on that day) per currency. The scenarios keep
  their shape but reflect the current vol level. The EWMA vol is one step ahead, so
  there's no look-ahead, and there's a test for that too.
- Parametric: VaR = z * sigma with an EWMA covariance (RiskMetrics, lambda 0.94, zero
  mean). ES = sigma * pdf(z) / (1 - confidence).

Multi-day figures use sqrt(h) scaling of the 1-day numbers.

Stressed VaR is historical VaR of today's book over the worst 250-day window in the data
we have, in the spirit of the Basel stressed VaR.

### Decomposition

- Standalone VaR: the trade on its own.
- Component VaR, an Euler allocation, so the components add up to portfolio VaR:
  - parametric: z * (e_i . Cov x) / sigma_p, exact.
  - historical: the textbook approach takes each trade's P&L on the single VaR day, which
    is very noisy. I average over the scenarios either side of the VaR quantile
    (+/- max(1, N/200) order statistics) and rescale so they add up to VaR.
- Marginal VaR: dVaR/dnotional = component VaR / signed notional, shown as the VaR change
  from buying another 1m of the base currency.
- Incremental VaR: VaR minus VaR without the trade, fully recomputed. It answers "what if
  I close this?".
- Component ES: each trade's average P&L on the days beyond VaR.
- Strategy-level risk runs the same engine on exposures aggregated by strategy, so
  offsetting trades inside a strategy net off first.

### Hedging

The best-hedges report follows Litterman's "Hot Spots and Hedges" (1996). For each
currency c, the hedge that minimises portfolio variance is a USD exposure of
-cov(c, book) / var(c). Every currency is a candidate, not just the ones held, so proxy
hedges show up (e.g. NZD for an AUD position). The trade is then turned into a spot trade
in the USD pair, rounded to 100k, and VaR is recomputed with the chosen method rather
than trusting the variance approximation.

### Backtest

For each of the last 250 days, VaR is computed using only the earlier window and compared
with that day's hypothetical P&L (today's exposures times that day's moves).

- Kupiec checks whether the number of exceptions is right.
- Christoffersen checks whether they cluster. Clustering means the model is slow to pick
  up vol changes, which is what plain historical simulation tends to do.
- The Basel traffic light is shown at 99%.

## Main assumptions

- Spot only, constant notional, no funding, carry, forward points or settlement.
- P&L is unrealised and in USD at today's rate. No closed trades and no costs.
- Prices are Yahoo composite mids, not executable quotes. The close is taken at midnight
  London as explained above.
- VaR uses sqrt(h) scaling and today's exposures, so no trading during the horizon.
- The sample book is made up. Its entry prices are real closes on each trade date with a
  few pips of fake slippage.

## Limitations and known issues

- Yahoo data is free and unofficial. The spike filter catches isolated bad prints but not
  sustained ones, so it isn't good enough for official P&L.
- Yahoo sometimes throttles Streamlit Cloud's shared IPs. When that happens the app uses
  the snapshot and marks can be stale. The alerts panel says so.
- With 500 scenarios the 99% tail is only about 5 observations, so historical VaR and
  especially historical component VaR are noisy. The averaging window, ES and the other
  two models help put it in context.
- sqrt(h) scaling ignores vol clustering and autocorrelation over longer horizons.
- Three years of history doesn't include 2008, the 2015 SNB de-peg or the 2016 Brexit
  vote, so stressed VaR is only "stressed" relative to the last three years. The
  hypothetical scenarios cover bigger moves.
- The best-hedge sizing is a variance (second moment) calculation. VaR after the hedge is
  recomputed properly, but the size isn't the exact VaR-minimising one for historical
  VaR.
- Only USD is supported as a reporting currency. Adding a currency is one line in
  `fxdash/market.py`.
- What-if edits live in the browser session only, nothing is saved.
- A few interface details Streamlit doesn't offer directly (the pinned header and tabs,
  dropdowns closing on a second click, pinch-only chart zoom) are done with a little CSS
  and a small page script at the bottom of `app.py`. They depend on Streamlit's page
  structure, so a Streamlit upgrade could need them adjusting.
- Local holidays are forward filled onto a common calendar, which slightly understates vol
  around them.

## What I'd do with more time

1. More products: FX forwards and NDFs (forward points, discounting, carry and roll-down
   P&L), then vanilla options with Garman-Kohlhagen pricing, greeks, and full revaluation
   VaR including vol surface moves.
2. Better data: a proper feed (Bloomberg / Refinitiv) with NY 5pm closes, streaming
   marks, and a time series store instead of a CSV. Longer history so stressed VaR can
   use 2008 or 2020.
3. Persistence: a trade database with an audit trail, realised P&L on closes, end of day
   P&L snapshots, and a full P&L explain (spot, carry, FX translation, new trades).
4. Risk: a proper ES backtest (Acerbi-Szekely), Monte Carlo VaR for an options book,
   factor views (USD beta, risk on/off beta), and liquidity-adjusted VaR using time to
   exit per currency.
5. PM workflow: limits per strategy and per currency with breach history, a hedge
   optimiser across several currencies with transaction costs, and a scheduled morning
   risk email.
6. Engineering: authentication, CI running the tests, typed config, and caching the
   heavier calculations for large books.

## Project layout

    app.py                        Streamlit app
    fxdash/market.py              Yahoo download, date fix, live marks, snapshot fallback
    fxdash/portfolio.py           loading and validation, valuation, P&L, exposures
    fxdash/risk.py                VaR / ES (3 methods), decomposition, hedges, backtest, stress
    data/portfolio.csv            sample book (13 trades, 12 pairs including crosses)
    data/usd_rates_snapshot.csv   3-year offline price snapshot
    .streamlit/config.toml        theme (colours, font, sizes)
    scripts/make_sample_data.py   rebuild the snapshot and sample book
    tests/test_calcs.py           unit tests
