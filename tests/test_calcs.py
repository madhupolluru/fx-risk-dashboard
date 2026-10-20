import numpy as np
import pandas as pd
import pytest

from fxdash.market import clean_spikes, pair_price, relabel_daily_bars, usd_pair
from fxdash.portfolio import (PortfolioError, daily_pnl_attribution, fill_at_market,
                              load_portfolio, net_by_pair, parse_ticket, usd_exposures,
                              value_positions)
from fxdash.risk import (backtest, best_hedges, christoffersen, compute_risk, daily_returns,
                         ewma_vol, fhs_scenarios, hedge_trade, hist_var, kupiec_pof,
                         stress_pnl, stressed_var)


def make_rates(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    vols = {"EUR": 0.006, "GBP": 0.007, "JPY": 0.008}
    start = {"EUR": 1.10, "GBP": 1.30, "JPY": 1 / 150}
    shocks = rng.multivariate_normal(np.zeros(3), [[1, .6, .2], [.6, 1, .1], [.2, .1, 1]], n)
    df = pd.DataFrame(index=idx)
    for i, c in enumerate(vols):
        df[c] = start[c] * np.exp(np.cumsum(shocks[:, i] * vols[c]))
    df.insert(0, "USD", 1.0)
    return df


def book(rows, **kw):
    return load_portfolio(pd.DataFrame(rows, columns=[
        "trade_id", "trade_date", "ccy_pair", "side", "notional", "entry_price"]), **kw)


def returns_of(rates):
    return daily_returns(rates, [c for c in rates if c != "USD"])


# P&L

def test_pnl_usd_quote_and_non_usd_quote():
    rates = make_rates()
    today = rates.iloc[-1]
    eur, jpy_px = today["EUR"], 1 / today["JPY"]
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.05),
               ("B", "2025-01-02", "USDJPY", "SELL", 1e6, 140.0)])
    v = value_positions(pf, rates)
    assert v.loc["A", "pnl_inception_usd"] == pytest.approx(1e6 * (eur - 1.05))
    # short USDJPY: JPY P&L is -1m * (S - K), translated to USD at today's rate
    assert v.loc["B", "pnl_inception_usd"] == pytest.approx(-1e6 * (jpy_px - 140.0) / jpy_px)
    assert v.loc["B", "pnl_quote"] == pytest.approx(-1e6 * (jpy_px - 140.0))


def test_daily_pnl_and_new_trade():
    rates = make_rates()
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.05),
               ("N", rates.index[-1].date().isoformat(), "EURUSD", "SELL", 1e6, 1.0)])
    v = value_positions(pf, rates)
    eur = rates["EUR"]
    assert v.loc["A", "pnl_daily_usd"] == pytest.approx(1e6 * (eur.iloc[-1] - eur.iloc[-2]))
    assert v.loc["N", "pnl_daily_usd"] == pytest.approx(v.loc["N", "pnl_inception_usd"])


def test_daily_attribution_adds_up_exactly():
    rates = make_rates()
    pf = book([("A", "2025-01-02", "EURGBP", "BUY", 3e6, 0.85),
               ("B", "2025-02-03", "USDJPY", "SELL", 2e6, 150.0),
               ("C", "2025-03-03", "GBPJPY", "BUY", 1e6, 190.0),
               ("N", rates.index[-1].date().isoformat(), "EURUSD", "SELL", 1e6, 1.0)])
    attr = daily_pnl_attribution(pf, rates)
    assert "New trades" in attr
    assert attr.sum() == pytest.approx(value_positions(pf, rates)["pnl_daily_usd"].sum())


def test_exposure_revaluation_is_exact_for_cross():
    rates = make_rates()
    pf = book([("X", "2025-01-02", "EURGBP", "BUY", 2e6, 0.85)])
    today = rates.iloc[-1]
    E = usd_exposures(pf, today)
    v0 = 2e6 * (today["EUR"] - 0.85 * today["GBP"])
    v1 = 2e6 * (today["EUR"] * 1.03 - 0.85 * today["GBP"] * 0.98)
    assert stress_pnl(E, {"EUR": 0.03, "GBP": -0.02}).iloc[0] == pytest.approx(v1 - v0)


def test_net_by_pair_breakeven():
    rates = make_rates()
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 3e6, 1.10),
               ("B", "2025-01-03", "EURUSD", "SELL", 1e6, 1.20)])
    net = net_by_pair(value_positions(pf, rates)).loc["EURUSD"]
    assert net["net_notional"] == pytest.approx(2e6)
    be = net["breakeven"]
    # at the breakeven the two trades' quote P&L cancels out
    assert 3e6 * (be - 1.10) - 1e6 * (be - 1.20) == pytest.approx(0, abs=1e-6)


# risk

@pytest.mark.parametrize("method", ["hist", "fhs", "param"])
def test_components_sum_to_var_and_flat_book(method):
    rates = make_rates()
    R = returns_of(rates)
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 5e6, 1.1),
               ("B", "2025-01-02", "GBPUSD", "SELL", 3e6, 1.3),
               ("C", "2025-01-02", "USDJPY", "BUY", 4e6, 150.0)])
    E = usd_exposures(pf, rates.iloc[-1])
    res = compute_risk(method, E, R, 0.99, 1, 250, pf["signed_notional"])
    assert res.positions["component_var"].sum() == pytest.approx(res.var)
    assert res.var <= res.positions["standalone_var"].sum() + 1e-6

    flat = book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.1),
                 ("B", "2025-01-02", "EURUSD", "SELL", 1e6, 1.1)])
    Ef = usd_exposures(flat, rates.iloc[-1])
    res = compute_risk(method, Ef, R, 0.99, 1, 250, flat["signed_notional"])
    assert res.var == pytest.approx(0, abs=1e-6)
    # no currency exposure at all shouldn't blow up either
    empty = compute_risk(method, Ef.iloc[:, :0], R, 0.99, 1, 250, flat["signed_notional"])
    assert empty.var == 0 and len(empty.positions) == 2


def test_parametric_matches_closed_form_single_position():
    rates = make_rates()
    R = returns_of(rates)
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.1)])
    E = usd_exposures(pf, rates.iloc[-1])
    res = compute_risk("param", E, R, 0.99, 10, 250, pf["signed_notional"])
    w = 0.94 ** np.arange(250 - 1, -1, -1)
    sigma = np.sqrt((w / w.sum() * R["EUR"].iloc[-250:] ** 2).sum())
    assert res.var == pytest.approx(2.326348 * sigma * abs(E.iloc[0, 0]) * np.sqrt(10), rel=1e-5)


def test_hist_var_quantile():
    pnl = -np.arange(100.0)  # losses 0..99
    assert hist_var(pnl, 0.99) == pytest.approx(98.01)


def test_ewma_vol_has_no_lookahead():
    R = returns_of(make_rates())
    sig, _ = ewma_vol(R)
    bumped = R.copy()
    bumped.iloc[150] *= 10
    sig2, _ = ewma_vol(bumped)
    # a shock on day 150 can only change the forecasts from day 151 onwards
    pd.testing.assert_frame_equal(sig.iloc[:151], sig2.iloc[:151])
    assert (sig2.iloc[151] > sig.iloc[151]).all()


def test_fhs_rescales_to_current_vol():
    R = returns_of(make_rates(600))
    R.iloc[-60:] *= 3  # vol regime shift at the end
    _, sig_next = ewma_vol(R)
    S = fhs_scenarios(R, 500)
    assert len(S) == 500
    # the scaled scenarios should look like today's vol, not the 500 day average
    assert (S.std() > R.iloc[-500:].std()).all()
    assert S.std().to_numpy() == pytest.approx(sig_next.to_numpy(), rel=0.35)


def test_stressed_var_at_least_as_big_as_any_window():
    rates = make_rates(600)
    R = returns_of(rates)
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 5e6, 1.1)])
    E = usd_exposures(pf, rates.iloc[-1])
    svar, start, end = stressed_var(E, R, 0.99, 1)
    assert svar >= compute_risk("hist", E, R, 0.99, 1, 250).var - 1e-6
    assert len(R.loc[start:end]) == 250


def test_best_hedge_reduces_var_and_trade_direction():
    rates = make_rates()
    R = returns_of(rates)
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 10e6, 1.1)])
    E = usd_exposures(pf, rates.iloc[-1])
    hedges = best_hedges(E, R, rates.iloc[-1], "param", 0.99, 1, 250)
    top = hedges.iloc[0]
    # the best hedge for long EUR is selling EUR itself, which takes VaR to ~0
    assert top["ccy"] == "EUR" and top["side"] == "SELL" and top["pair"] == "EURUSD"
    assert top["var_after"] == pytest.approx(0, abs=1e-6)
    # GBP is correlated with EUR so it should also help, just less
    gbp = hedges.set_index("ccy").loc["GBP"]
    assert 0 < gbp["pct_reduction"] < 1 and gbp["side"] == "SELL"


def test_hedge_trade_conventions():
    row = pd.Series({"USD": 1.0, "EUR": 1.10, "JPY": 1 / 150})
    assert usd_pair("EUR") == "EURUSD" and usd_pair("JPY") == "USDJPY"
    assert hedge_trade("EUR", 11e6, row) == ("EURUSD", "BUY", 10e6)
    # long JPY exposure means selling USDJPY
    assert hedge_trade("JPY", 5e6, row) == ("USDJPY", "SELL", 5e6)
    assert hedge_trade("JPY", -5e6, row) == ("USDJPY", "BUY", 5e6)


# backtesting

def test_kupiec():
    assert kupiec_pof(3, 250, 0.99) > 0.5
    assert kupiec_pof(15, 250, 0.99) < 0.01


def test_christoffersen_flags_clustering():
    spread = np.zeros(250, dtype=bool)
    spread[[20, 80, 140, 200]] = True
    clustered = np.zeros(250, dtype=bool)
    clustered[100:104] = True
    assert christoffersen(spread)[1] > 0.5
    assert christoffersen(clustered)[1] < 0.01


@pytest.mark.parametrize("method", ["hist", "fhs", "param"])
def test_backtest_shape(method):
    rates = make_rates(400)
    pf = book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.1)])
    E = usd_exposures(pf, rates.iloc[-1])
    bt = backtest(E, returns_of(rates), method, 0.99, lookback=250, n_test=100)
    assert len(bt) == 100 and (bt["var"] > 0).all()


# loading and data cleaning

def test_validation_errors():
    with pytest.raises(PortfolioError):
        book([("A", "2025-01-02", "EURXYZ", "BUY", 1e6, 1.1)])
    with pytest.raises(PortfolioError):
        book([("A", "2025-01-02", "EURUSD", "HOLD", 1e6, 1.1)])
    with pytest.raises(PortfolioError, match="after the latest market date"):
        book([("A", "2030-01-02", "EURUSD", "BUY", 1e6, 1.1)], asof=pd.Timestamp("2026-01-01"))
    with pytest.raises(PortfolioError, match="duplicate"):
        book([("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.1),
              ("A", "2025-01-02", "EURUSD", "BUY", 1e6, 1.1)])


def test_loader_is_forgiving_about_format():
    pf = book([("A", "2025-01-02", "eur/usd", "long", "1,000,000", "1.1"),
               (None, "2025-01-02", "USD-JPY", "s", 2e6, 150)])
    assert list(pf["ccy_pair"]) == ["EURUSD", "USDJPY"]
    assert list(pf["side"]) == ["BUY", "SELL"]
    assert pf["notional"].iloc[0] == 1e6
    assert pf.index[1] == "NEW1"


def test_fill_at_market():
    rates = make_rates()
    raw = pd.DataFrame([{"trade_id": "N", "trade_date": None, "ccy_pair": "EURUSD",
                         "side": "BUY", "notional": 1e6, "entry_price": None}])
    filled = fill_at_market(raw, rates)
    assert filled.loc[0, "trade_date"] == rates.index[-1]
    assert filled.loc[0, "entry_price"] == pytest.approx(rates["EUR"].iloc[-1], abs=1e-6)
    pf = load_portfolio(filled)
    assert value_positions(pf, rates).loc["N", "pnl_inception_usd"] == pytest.approx(0, abs=1)


def test_relabel_daily_bars_moves_back_one_business_day():
    idx = pd.to_datetime(["2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07"])
    df = pd.DataFrame({"EUR": [1.0874, 1.0874, 1.0934, 1.0733]}, index=idx)
    out = relabel_daily_bars(df)
    # the 7 Nov bar opened at 1.0733, i.e. after the election selloff on the 6th
    assert out.loc["2024-11-06", "EUR"] == pytest.approx(1.0733)
    assert out.index[0] == pd.Timestamp("2024-11-01")  # Monday's bar -> Friday's close


def test_spike_cleaning_keeps_last_day():
    s = pd.DataFrame({"EUR": [1.1] * 10 + [1.5] + [1.1] * 5 + [1.3]},
                     index=pd.bdate_range("2025-01-01", periods=17))
    c = clean_spikes(s)
    assert c["EUR"].iloc[10] == pytest.approx(1.1)
    assert c["EUR"].iloc[-1] == pytest.approx(1.3)
    assert pair_price("EURUSD", pd.Series({"EUR": 1.1, "USD": 1.0})) == pytest.approx(1.1)


def test_parse_ticket():
    assert parse_ticket("BUY 25m EURUSD") == {"side": "BUY", "notional": 25e6,
                                              "ccy_pair": "EURUSD", "entry_price": None}
    t = parse_ticket("sell 5,000,000 usd/jpy @ 151.2")
    assert (t["side"], t["notional"], t["ccy_pair"]) == ("SELL", 5e6, "USDJPY")
    assert t["entry_price"] == 151.2
    assert parse_ticket("B 1.5bn AUD-NZD")["notional"] == 1.5e9
    # the hedge labels in the dropdown parse as well
    assert parse_ticket("SELL 64.7m AUDUSD · VaR $352k")["notional"] == pytest.approx(64.7e6)
    for bad in ["BUY EURUSD", "HOLD 10m EURUSD", "BUY 10m EURXYZ", "BUY 10m EUREUR"]:
        with pytest.raises(PortfolioError):
            parse_ticket(bad)
