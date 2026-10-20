# FX portfolio risk dashboard.
# run with:  streamlit run app.py

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

from fxdash.market import LONDON, SUPPORTED_CCYS, get_market_data, pair_price, usd_pair
from fxdash.portfolio import (PortfolioError, daily_pnl_attribution, fill_at_market,
                              load_portfolio, net_by_pair, parse_ticket, sanity_warnings,
                              usd_exposures, value_history, value_positions)
from fxdash.risk import (HISTORICAL_EPISODES, STRESS_SCENARIOS, backtest, basel_zone,
                         best_hedges, christoffersen, compute_risk, daily_returns,
                         episode_moves, ewma_cov, kupiec_pof, scenario_pnl, scenarios_for,
                         stress_pnl, stressed_var, var_of, worst_window)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "portfolio.csv"
EDIT_COLS = ["trade_id", "trade_date", "ccy_pair", "side", "notional", "entry_price",
             "strategy"]

# palette, kept in step with .streamlit/config.toml. Jade / coral only ever mean gain / loss
# (or risk reduced / added), steel blue is the one accent and the neutral data colour, and
# strategies get the categorical colours in a fixed order (no green or red among them).
INK, MUTED, GRID = "#1c2433", "#8a93a3", "#e9ecf1"
BLUE, BLUE_LIGHT, BLUE_PALE = "#3d6db3", "#aec3e3", "#dbe5f3"
GAIN, LOSS, LOSS_DARK = "#2b9a73", "#d4604f", "#a8473a"
GAIN_TEXT, LOSS_TEXT = "#1f7d5c", "#b84a3a"  # darker steps so coloured numbers stay readable
SERIES = ["#3d6db3", "#e07b39", "#7a5cc0", "#d9a21b", "#1898b8"]
AMBER, AMBER_TEXT = "#f2a516", "#c97a00"
STATUS = {"red": LOSS_TEXT, "orange": AMBER_TEXT, "blue": "#5b6474", "green": GAIN_TEXT}
METHOD_OPTIONS = {
    "Historical simulation": "hist",
    "Filtered historical (vol-scaled)": "fhs",
    "Parametric (EWMA)": "param",
}

st.set_page_config(page_title="FX Portfolio Risk", page_icon="💱", layout="wide")
st.markdown("""
<style>
/* top padding chosen so the header rests exactly where it gets pinned (3.75rem, just under
   Streamlit's toolbar). It then doesn't move at all when you start scrolling. */
.block-container {padding-top: 2.75rem; padding-bottom: 3rem; max-width: 1560px;}

/* header */
.hero {margin: 0 0 14px;}
.hero h1 {color: #1c2433; font-size: 26px; font-weight: 700; margin: 0; padding: 0;
          letter-spacing: -0.01em;}
.hero .sub {color: #5b6474; font-size: 13px; margin-top: 2px;}
.chip {display: inline-block; padding: 2px 10px; margin: 8px 6px 0 0; border-radius: 6px;
       font-size: 12px; font-weight: 500; color: #3b4558; background: #eceff4;
       border: 1px solid #dde2ea;}

/* cards */
[data-testid="stMetric"] {background: #fff; border: 1px solid #dde2ea !important;
    border-radius: 12px; padding: 0 !important; box-shadow: 0 1px 2px rgba(16,33,58,0.05);}
[data-testid="stMetric"] > div {padding: 10px 12px !important;}
[data-testid="stMetricLabel"] p {font-size: 12px; font-weight: 500; color: #5b6474;}
[data-testid="stMetricValue"] {font-size: 20px; font-weight: 600; color: #1c2433;}
[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #dde2ea; border-radius: 12px;
    overflow: hidden; box-shadow: 0 1px 2px rgba(16,33,58,0.05);}
[data-testid="stExpander"] details {background: #fff;}

h5 {font-size: 16px !important; font-weight: 600 !important; color: #1c2433;
    margin: 12px 0 6px !important; padding: 0 !important;}
[data-testid="stCaptionContainer"] {font-size: 12.5px;}
[data-testid="stWidgetLabel"] p {font-size: 13px;}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p,
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] p {font-size: 12px !important;}
[data-testid="stButtonGroup"] button p {font-size: 13px !important;}
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] p,
[data-testid="stTooltipContent"] li {font-size: 13px !important;}
.stTabs [data-baseweb="tab"] p {font-size: 14px;}
[data-testid="stSidebar"] h5 {margin-top: 18px !important;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 2px;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; font-weight: 600; color: #5b6474;}
.stTabs [aria-selected="true"] {color: #3d6db3;}

/* pinned header, just under Streamlit's toolbar. The sticky rule has to go on Streamlit's
   wrapper, the container inside it is exactly as tall as the wrapper so it can't stick. */
[data-testid="stLayoutWrapper"]:has(> .st-key-top_bar) {position: sticky; top: 3.75rem;
    z-index: 990; background: #f6f7f9; padding-bottom: 26px;}

/* the tab bar is pinned right under the header so the two act as one block. The negative
   margin cancels the usual gap above the tabs so the spacing is the same before and after
   it pins. The header's height changes (what-if banner, narrow screens), so the page
   script keeps --fx-top-h in sync, and the 1px overlap stops a seam from rounding. */
[data-testid="stTabs"] {margin-top: calc(-1rem - 1px);}
[data-testid="stTabs"] div:has(> [role="tablist"]) {position: sticky;
    top: calc(3.75rem + var(--fx-top-h, 244px) - 1px); z-index: 989; background: #f6f7f9;
    border-bottom: 1px solid #e4e8ee; box-shadow: 0 6px 8px -8px rgba(28,36,51,0.18);}

/* Exit what-if: same height as the banner next to it, at any width. The row is stretched
   and every wrapper down to the button fills it (the tooltip adds a couple of spans). */
[data-testid="stHorizontalBlock"]:has(.st-key-exit_whatif_top) {align-items: stretch;}
[data-testid="stHorizontalBlock"]:has(.st-key-exit_whatif_top) > [data-testid="stColumn"]
  > [data-testid="stVerticalBlock"] {height: 100%;}
.st-key-exit_whatif_top, .st-key-exit_whatif_top .stButton,
.st-key-exit_whatif_top .stButton > div, .st-key-exit_whatif_top .stTooltipIcon,
.st-key-exit_whatif_top .stTooltipHoverTarget, .st-key-exit_whatif_top button {height: 100%;}
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > .st-key-exit_whatif_top)
  [data-testid="stAlert"] {height: 100%;}

/* Remove buttons on the what-if trades: soft coral so they read as a removal action */
[class*="st-key-remove_"] button {color: #b84a3a; background: #fbefec; border-color: #efc6bd;}
[class*="st-key-remove_"] button:hover {color: #fff; background: #d4604f; border-color: #d4604f;}
/* keep the trade text level with its Remove button. Streamlit pairs the paragraph's bottom
   margin with a negative one on its container, so both get reset. */
[data-testid="stHorizontalBlock"]:has([class*="st-key-remove_"])
  [data-testid="stMarkdownContainer"],
[data-testid="stHorizontalBlock"]:has([class*="st-key-remove_"])
  [data-testid="stMarkdownContainer"] p {margin-bottom: 0;}

/* the explanation under a chart is centred on it */
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"])
  + [data-testid="stElementContainer"] [data-testid="stCaptionContainer"] {text-align: center;}

/* reset hint in the chart's top-right corner, only while the pointer is over the chart,
   so it doesn't add a line of text under every chart */
[data-testid="stPlotlyChart"] {position: relative;}
[data-testid="stPlotlyChart"]::after {
    content: "Double-click to reset"; position: absolute; top: 8px; right: 12px;
    font-size: 12px; color: #8a93a3; background: rgba(255,255,255,0.92);
    padding: 2px 8px; border: 1px solid #e4e8ee; border-radius: 6px;
    opacity: 0; transition: opacity 0.15s; pointer-events: none;}
[data-testid="stPlotlyChart"]:hover::after {opacity: 1;}

/* limit usage bar */
.limit {margin: 4px 0 6px;}
.limit .row {display: flex; justify-content: space-between; font-size: 13px;
             color: #5b6474; margin-bottom: 5px;}
.limit .row b {color: #1c2433;}
.limit .track {position: relative; height: 10px; background: #e3e8f0; border-radius: 999px;}
.limit .fill {height: 10px; border-radius: 999px;}
.limit .tick {position: absolute; top: -3px; width: 2px; height: 16px; background: #8a93a3;}

/* alerts list */
.alerts {background: #fff; border: 1px solid #dde2ea; border-radius: 12px; padding: 6px 4px;}
.alert {display: flex; gap: 10px; align-items: baseline; padding: 7px 12px;
        font-size: 13px; color: #1c2433; line-height: 1.45;}
.alert + .alert {border-top: 1px solid #eef1f6;}
.alert .tag {flex: 0 0 auto; min-width: 92px; text-align: center; font-size: 11px;
             font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
             padding: 2px 8px; border-radius: 6px;}
</style>
""", unsafe_allow_html=True)


# --- small helpers ---

def fmt_usd(x, signed=False):
    if x is None or not np.isfinite(x):
        return "-"
    sign = "+" if signed and x > 0 else ("-" if x < 0 else "")
    a = abs(x)
    if a >= 1e9:
        body = f"{a / 1e9:,.2f}bn"
    elif a >= 1e6:
        body = f"{a / 1e6:,.2f}m"
    elif a >= 1e3:
        body = f"{a / 1e3:,.0f}k"
    else:
        body = f"{a:,.0f}"
    return f"{sign}${body}"


def md_text(text: str) -> str:
    # Streamlit markdown reads text between two $ signs as a maths formula, so a sentence
    # with two dollar amounts ends up half in a maths font. Escape them in markdown text.
    return text.replace("$", r"\$")


def html_text(text: str) -> str:
    # same problem inside the raw HTML blocks, there the HTML code for $ does the job
    return html.escape(text).replace("$", "&#36;")


def pnl_color(v):
    if isinstance(v, (int, float)) and np.isfinite(v) and v != 0:
        return f"color: {GAIN_TEXT if v > 0 else LOSS_TEXT}"
    return ""


def tidy(fig, height=360, **layout):
    # one look for every chart: transparent background (the card is white), soft grid,
    # same font as the app, strategy colours in a fixed order
    # (no rounded bar corners, plotly can't draw them for charts on a hidden tab)
    fig.update_layout(height=height, margin=dict(l=14, r=18, t=38, b=12),
                      template="plotly_white", colorway=SERIES,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=12, color="#3b4558"),
                      hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font_color=INK),
                      legend=dict(orientation="h", y=1.1, x=0, title=None, font=dict(size=12)),
                      **layout)
    # sizes set explicitly, otherwise Streamlit's chart theme fills in its own (10.5, 12.25
    # and 14px). 11px tick labels, 12px for everything else, 13px chart titles.
    axis = dict(gridcolor=GRID, zerolinecolor="#c3cad6", linecolor=GRID,
                tickfont=dict(size=11), title_font=dict(size=12))
    fig.update_xaxes(**axis)
    # zoom and pan only move left and right, the vertical scale stays put
    fig.update_yaxes(**axis, fixedrange=True)
    fig.update_layout(dragmode="pan")
    fig.update_annotations(font_size=12)
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(size=13, color=INK))
    return fig


def split_at_zero(s: pd.Series):
    # Split a series into its positive and negative parts for a two-colour line. A point is
    # added exactly where it crosses zero (interpolated between the two days) so the green
    # and red pieces join up instead of leaving a gap at each crossing.
    x = s.index.asi8.astype(float)
    y = s.to_numpy(dtype=float)
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(y)):
        if y[i - 1] * y[i] < 0:
            xs.append(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
            ys.append(0.0)
        xs.append(x[i])
        ys.append(y[i])
    full = pd.Series(ys, index=pd.to_datetime(np.array(xs, dtype="int64")))
    return full.where(full >= 0), full.where(full <= 0)


# no toolbar: drag to move along, pinch (or Ctrl + scroll) to zoom, double-click to reset.
# A plain scroll still scrolls the page, see the page script at the bottom of the file.
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}


def show_chart(fig):
    st.plotly_chart(fig, config=CHART_CONFIG)


def hbar(values: pd.Series, height=360, title=None, text=None, risk=False):
    # horizontal bar coloured by sign. For P&L positive is green, for risk numbers
    # (risk=True) positive means more risk so it's red.
    good = (values < 0) if risk else (values >= 0)
    fig = go.Figure(go.Bar(x=values.to_numpy(), y=values.index.astype(str), orientation="h",
                           marker_color=[GAIN if g else LOSS for g in good],
                           text=text, textposition="auto", textangle=0,
                           insidetextfont=dict(color="white")))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left"))
    return tidy(fig, height, xaxis_title="USD", showlegend=False)


def fit(n_rows: int) -> int:
    # height that shows every row, so tables never get their own scrollbar. A table that
    # can scroll grabs the mouse wheel and the page stops scrolling under the pointer.
    # rows are 31px at the 14px base font set in config.toml
    return 31 * (n_rows + 1) + 4


def pick_one(label_text, options, key, **kw):
    # chip buttons for short lists. Clicking the active chip would clear it, so the last
    # choice gets put back if that happens.
    options = list(options)
    last = f"{key}_last"
    if st.session_state.get(key) not in options:
        previous = st.session_state.get(last)
        st.session_state[key] = previous if previous in options else options[0]

    def keep():
        if st.session_state[key] is None:
            st.session_state[key] = st.session_state.get(last, options[0])

    value = st.pills(label_text, options, key=key, on_change=keep, **kw)
    st.session_state[last] = value
    return value


def label(df):
    return df.index.astype(str) + " " + df["ccy_pair"].to_numpy()


def ratio_color(v):
    # for EWMA / long run vol: red when vol is running hot, green when it's quiet
    if not np.isfinite(v):
        return ""
    if v > 1.3:
        return "background-color: #f7e2dd"
    if v < 0.8:
        return "background-color: #dcefe7"
    return ""


@st.cache_data(ttl=600, show_spinner="Loading FX prices...")
def load_market(prefer_live: bool, bust: int):
    return get_market_data(years=3, prefer_live=prefer_live)


def editable_book(pf: pd.DataFrame) -> pd.DataFrame:
    df = pf.reset_index()[EDIT_COLS].copy()
    df["trade_date"] = df["trade_date"].dt.date
    return df


def book_key(p: pd.DataFrame) -> str:
    # used to tell whether the edited book differs from the original
    cols = ["trade_date", "ccy_pair", "side", "notional", "entry_price", "strategy"]
    p = p[cols].sort_index().astype({"notional": float, "entry_price": float})
    return p.to_csv()


def reset_book():
    st.session_state["book"] = st.session_state["base_book"].copy()
    st.session_state["ed_ver"] += 1
    st.session_state["flash"] = "Back to the original book."


def remove_trade(trade_id: str):
    # drop one trade from the what-if book (used by the Remove buttons on the Hedging tab)
    book = st.session_state.get("book_live", st.session_state["book"])
    st.session_state["book"] = book[book["trade_id"].astype(str) != trade_id] \
        .reset_index(drop=True)
    st.session_state["ed_ver"] += 1
    st.session_state["flash"] = f"Removed {trade_id} from the what-if book."


def in_book(row: dict, check_price: bool = True) -> bool:
    # is this trade already in the what-if book? Same pair, side, notional and date, and the
    # same price when one was given. Trades booked at market skip the price check, spot
    # moves every time the prices refresh and it would still be the same trade.
    b = st.session_state.get("book_live")
    if b is None or len(b) == 0:
        return False
    pairs = b["ccy_pair"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    dates = pd.to_datetime(b["trade_date"], errors="coerce").dt.normalize()
    same = ((pairs == row["ccy_pair"])
            & (b["side"].astype(str).str.upper() == row["side"])
            & np.isclose(pd.to_numeric(b["notional"], errors="coerce"), row["notional"],
                         rtol=1e-9)
            & (dates == pd.Timestamp(row["trade_date"])))
    if check_price:
        same &= np.isclose(pd.to_numeric(b["entry_price"], errors="coerce"),
                           row["entry_price"], rtol=1e-7)
    return bool(same.any())


def add_trade(row: dict, hedge_label: str | None = None):
    # called from the hedging tab. Appends to whatever is in the editor right now.
    # Hedges are remembered so they stay in the dropdown (greyed out) after the hedge list
    # is recalculated for the new book.
    if hedge_label:
        st.session_state.setdefault("added_hedges", {})[hedge_label] = row
    book = st.session_state.get("book_live", st.session_state["book"])
    st.session_state["book"] = pd.concat([book, pd.DataFrame([row])], ignore_index=True)
    st.session_state["ed_ver"] += 1
    st.session_state["flash"] = f"Added {row['side']} {row['notional']:,.0f} {row['ccy_pair']} " \
                                "to the what-if book."


# --- sidebar ---

with st.sidebar:
    st.markdown("##### Portfolio")
    src = st.radio("Source", ["Sample portfolio", "Upload CSV"], label_visibility="collapsed")
    upload = None
    if src == "Upload CSV":
        upload = st.file_uploader("Portfolio CSV", type="csv")
        st.caption("Columns: trade_id, trade_date, ccy_pair, side (BUY/SELL), notional "
                   "(base ccy units), entry_price, strategy (optional). "
                   f"Currencies: {', '.join(SUPPORTED_CCYS)}.")
    st.download_button("Download sample CSV", SAMPLE_PATH.read_bytes(),
                       file_name="portfolio.csv", mime="text/csv")

    st.markdown("##### Risk settings")
    method_label = st.selectbox(
        "VaR method", list(METHOD_OPTIONS), key="method",
        help="- **Historical**: uses the raw past moves.\n"
             "- **Filtered historical**: rescales them to today's EWMA volatility.\n"
             "- **Parametric**: assumes normal moves with an EWMA covariance.")
    method = METHOD_OPTIONS[method_label]
    # any whole number in a sensible range. 750 days fits inside the ~3 years of prices
    # every data source has, and still leaves the backtest a full window.
    alpha = st.slider("Confidence (%)", min_value=90, max_value=99, value=99, step=1) / 100
    horizon = st.slider("Horizon (days)", min_value=1, max_value=20, value=1, step=1)
    lookback = st.slider("Lookback (trading days)", min_value=100, max_value=750, value=500,
                         step=1, help="Use the arrow keys for single-day steps.")
    limit = st.number_input("VaR limit (USD)", min_value=0, value=750_000, step=50_000,
                            help="Applied to VaR at the confidence and horizon above.")

    st.markdown("##### Market data")
    live = st.toggle("Live Yahoo Finance prices", value=True)
    if st.button("Refresh prices"):
        st.session_state["bust"] = st.session_state.get("bust", 0) + 1

    st.markdown("##### Charts")
    st.caption("Drag to move along. Pinch (or Ctrl + scroll) to zoom in and out. "
               "Double-click to reset.")

md = load_market(live, st.session_state.get("bust", 0))
rates = md.rates
asof = md.asof
ccys = [c for c in rates.columns if c != "USD"]
returns = daily_returns(rates, ccys)

if upload is not None:
    raw_src, src_id = upload.getvalue(), hashlib.md5(upload.getvalue()).hexdigest()
else:
    raw_src, src_id = SAMPLE_PATH, "sample"

try:
    base_pf = load_portfolio(raw_src, asof=asof)
except PortfolioError as e:
    st.error(f"Could not load the portfolio: {e}")
    st.stop()

# the what-if editor works on a copy of the book kept in session state. It is reset
# whenever the source file changes.
if st.session_state.get("book_src") != src_id:
    st.session_state["book_src"] = src_id
    st.session_state["base_book"] = editable_book(base_pf)
    st.session_state["book"] = st.session_state["base_book"].copy()
    st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1


# --- header ---

# title, tags, KPI cards and the limit bar stay pinned to the top while the tabs scroll
top_bar = st.container(key="top_bar")
# The title is filled in further down, once we know whether the book has been edited. On a
# rerun Streamlit empties the placeholder straight away, which made the header collapse for
# a moment and the tabs jump, so the last run's title goes back in until the new one is ready.
hero = top_bar.empty()
if "hero_html" in st.session_state:
    hero.markdown(st.session_state["hero_html"], unsafe_allow_html=True)
kpi_box = top_bar.container()
tabs = st.tabs(["Overview", "Positions", "Risk", "Hedging", "Stress", "P&L & backtest",
                "Market", "What-if"])


# --- what-if editor ---
# Rendered first (even though it's the last tab) so the edited book drives every other tab.

with tabs[7]:
    st.markdown("##### Edit, add or delete trades")
    st.caption("Every tab updates from this book, so it doubles as a pre-trade check. Leave "
               "trade_date blank for today and entry_price blank to book at the market close "
               "on the trade date. Trades added from the Hedging tab land here too.")
    if "flash" in st.session_state:
        # a toast shows on whatever tab you're on, the button may be on another tab
        st.toast(st.session_state.pop("flash"), icon="✅")
    edited = st.data_editor(
        st.session_state["book"], num_rows="dynamic", key=f"editor_{st.session_state['ed_ver']}",
        hide_index=True,
        height=fit(max(len(st.session_state["book"]),
                       len(st.session_state.get("book_live", []))) + 1),
        column_config={
            "trade_id": st.column_config.TextColumn("trade_id", help="Blank = auto"),
            "trade_date": st.column_config.DateColumn("trade_date"),
            "ccy_pair": st.column_config.TextColumn("ccy_pair", help="e.g. EURUSD, USDJPY, EURGBP"),
            "side": st.column_config.SelectboxColumn("side", options=["BUY", "SELL"],
                                                     required=True),
            "notional": st.column_config.NumberColumn("notional", format="%,.0f", min_value=0),
            "entry_price": st.column_config.NumberColumn("entry_price", format="%.5f"),
        })
    st.session_state["book_live"] = edited
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("Reset book", on_click=reset_book, width="stretch")
    c2.download_button("Download CSV", edited.to_csv(index=False).encode(),
                       file_name="portfolio_edited.csv", mime="text/csv", width="stretch")

try:
    pf = load_portfolio(fill_at_market(edited, rates), asof=asof)
except PortfolioError as e:
    with tabs[7]:
        st.error(f"The edited book has problems so the original is shown everywhere else: {e}")
    pf = base_pf
what_if = book_key(pf) != book_key(base_pf)


# --- core calculations ---

pos = value_positions(pf, rates)
E = usd_exposures(pf, rates.iloc[-1])
risk = compute_risk(method, E, returns, alpha, horizon, lookback, pf["signed_notional"])
pos = pos.join(risk.positions)
var_1d = risk.var / np.sqrt(horizon)

E_strat = E.groupby(pf["strategy"]).sum()
strat_risk = compute_risk(method, E_strat, returns, alpha, horizon, lookback)

svar, svar_start, svar_end = stressed_var(E, returns, alpha, horizon)
bt_lookback = min(lookback, 500)
bt = backtest(E, returns, method, alpha, lookback=bt_lookback, n_test=250)

# every method at two confidence levels, for the comparison table and the alerts
compare = {(m, a): compute_risk(m, E, returns, a, horizon, lookback)
           for m in METHOD_OPTIONS.values() for a in (0.95, 0.99)}

if what_if:
    E_base = usd_exposures(base_pf, rates.iloc[-1])
    base_risk = compute_risk(method, E_base, returns, alpha, horizon, lookback,
                             base_pf["signed_notional"])

# 1-day P&L of today's book on each historical day, used in a few places
x_net = E.sum(axis=0)
hist_book_pnl = returns[E.columns].iloc[-lookback:] @ x_net


# --- KPIs ---

if md.live and md.mark_time is not None:
    data_chip = f"Live · {md.mark_time.tz_convert(LONDON):%H:%M} London"
elif md.live:
    data_chip = "Live · last close"
else:
    data_chip = "Snapshot prices"
chips = [data_chip, method_label, f"{alpha:.0%} · {horizon}-day", f"{lookback}-day lookback",
         f"{len(pf)} trades"]
if what_if:
    chips.append("What-if book")
st.session_state["hero_html"] = (
    f"""<div class="hero"><h1>FX Portfolio Risk</h1>
    <div class="sub">Marks as of {asof:%A %d %B %Y} · prices from {html.escape(md.source)}</div>
    {"".join(f'<span class="chip">{html.escape(t)}</span>' for t in chips)}</div>""")
hero.markdown(st.session_state["hero_html"], unsafe_allow_html=True)


def limit_bar(util: float, limit: float) -> str:
    # status colour plus a word, so the state never relies on colour alone
    if util >= 1:
        fill, text, word = LOSS, LOSS_TEXT, "Limit breached"
    elif util >= 0.8:
        fill, text, word = AMBER, AMBER_TEXT, "Close to limit"
    else:
        fill, text, word = BLUE, BLUE, "Within limit"
    return f"""<div class="limit"><div class="row">
      <span>VaR limit usage · <b style="color:{text}">{word}</b></span>
      <span><b style="color:{text}">{util:.0%}</b> of {html_text(fmt_usd(limit))}</span></div>
      <div class="track"><div class="fill" style="width:{min(util, 1) * 100:.1f}%;
      background:{fill}"></div><div class="tick" style="left:80%"></div></div></div>"""


with kpi_box:
    # the banner always has a slot, empty outside what-if mode. Adding it as a new element
    # only when needed made the header briefly overshoot while Streamlit swapped elements.
    banner = st.empty()
    if what_if:
        c1, c2 = banner.container().columns([5, 1])  # CSS stretches the button to the banner
        c1.info("**What-if mode**: everything below reflects the edited book. The small "
                "numbers under each figure are the change vs the original book.", icon="🧪")
        c2.button("Exit what-if", on_click=reset_book, width="stretch", key="exit_whatif_top",
                  type="primary",
                  help="Drop all what-if changes and go back to the original book.")

    def delta(now, before, signed=True):
        return fmt_usd(now - before, signed) if what_if else None

    if what_if:
        base_pos = value_positions(base_pf, rates)
        base_svar = stressed_var(E_base, returns, alpha, horizon)[0]
    else:
        base_pos, base_svar = pos, svar
    k = st.columns(6)
    k[0].metric("Inception P&L", fmt_usd(pos["pnl_inception_usd"].sum(), True),
                delta(pos["pnl_inception_usd"].sum(), base_pos["pnl_inception_usd"].sum()),
                border=True)
    k[1].metric("Daily P&L", fmt_usd(pos["pnl_daily_usd"].sum(), True),
                delta(pos["pnl_daily_usd"].sum(), base_pos["pnl_daily_usd"].sum()), border=True,
                help="Change in USD value since the previous close (7pm NY).")
    k[2].metric("Gross notional", fmt_usd(pos["usd_notional"].abs().sum()),
                delta(pos["usd_notional"].abs().sum(), base_pos["usd_notional"].abs().sum()),
                delta_color="off", border=True)
    k[3].metric(f"VaR {alpha:.0%} · {horizon}d", fmt_usd(risk.var),
                delta(risk.var, base_risk.var if what_if else risk.var), delta_color="inverse",
                border=True)
    k[4].metric("ES", fmt_usd(risk.es),
                delta(risk.es, base_risk.es if what_if else risk.es), delta_color="inverse",
                border=True, help="Expected shortfall: average loss on the days worse than VaR.")
    # no help icon here, it squeezes the label on smaller screens. The window is in the Risk tab.
    k[5].metric("Stressed VaR", fmt_usd(svar), delta(svar, base_svar), delta_color="inverse",
                border=True)
    if limit > 0:
        st.markdown(limit_bar(risk.var / limit, limit), unsafe_allow_html=True)


# --- alerts ---

def build_alerts():
    # (colour, tag, message). Ordered roughly by how much a PM would care.
    out = []
    if limit > 0 and risk.var >= limit:
        out.append(("red", "Limit", f"VaR {fmt_usd(risk.var)} is over the {fmt_usd(limit)} "
                                    "limit."))
    elif limit > 0 and risk.var >= 0.8 * limit:
        out.append(("orange", "Limit", f"VaR is at {risk.var / limit:.0%} of the limit."))

    day = pos["pnl_daily_usd"].sum()
    if day < -var_1d:
        out.append(("red", "P&L", f"Today's loss of {fmt_usd(day)} is bigger than 1-day VaR "
                                  f"({fmt_usd(var_1d)})."))

    for n in md.notes:
        out.append(("orange", "Data", n))

    if not bt.empty:
        x = int(bt["exception"].sum())
        p_kup = kupiec_pof(x, len(bt), alpha)
        _, p_ind = christoffersen(bt["exception"])
        zone = basel_zone(x, len(bt), alpha)
        if zone in ("Amber", "Red") or p_kup < 0.05:
            out.append(("orange", "Model", f"{x} VaR exceptions in the last {len(bt)} days "
                                           f"(expected {len(bt) * (1 - alpha):.1f}), Kupiec p = "
                                           f"{p_kup:.2f}. The model looks too optimistic for "
                                           "this book."))
        if p_ind < 0.05:
            out.append(("orange", "Model", "VaR exceptions are clustered (Christoffersen "
                                           f"p = {p_ind:.2f}), the model reacts slowly to vol "
                                           "changes. Filtered historical should help."))

    gross_exp = x_net.abs().sum()
    if len(x_net):
        vol_now = np.sqrt(np.diag(ewma_cov(returns[E.columns].iloc[-lookback:])))
        vol_avg = returns[E.columns].iloc[-lookback:].std().to_numpy()
        ratio = pd.Series(vol_now / vol_avg, index=E.columns)
        hot = ratio[(ratio > 1.3) & (x_net.abs() > 0.1 * gross_exp)]
        for c, r in hot.items():
            out.append(("orange", "Vol", f"{c} vol is running at {r:.1f}x its lookback average "
                                         "and you have a material position, plain historical "
                                         "VaR probably understates the risk."))

    if risk.var > 0:
        top = pos["pct_of_var"].idxmax()
        share = pos.loc[top, "pct_of_var"]
        if share > 0.35:
            out.append(("blue", "Concentration", f"{top} {pos.loc[top, 'ccy_pair']} is "
                                                 f"{share:.0%} of portfolio VaR on its own."))
        ctop = risk.currencies["pct_of_var"].idxmax()
        cshare = risk.currencies.loc[ctop, "pct_of_var"]
        if cshare > 0.5:
            out.append(("blue", "Concentration", f"{ctop} drives {cshare:.0%} of VaR."))

    # pairs in the book that moved a lot today relative to their recent vol
    for pair in sorted(pf["ccy_pair"].unique()):
        lr = np.log(pair_price(pair, rates)).diff().dropna()
        sd = lr.iloc[-22:-1].std()
        if sd > 0 and abs(lr.iloc[-1]) > 2 * sd:
            out.append(("blue", "Market", f"{pair} moved {np.expm1(lr.iloc[-1]):+.2%} today, "
                                          f"{abs(lr.iloc[-1]) / sd:.1f}x its 1M daily vol."))

    vals = [compare[(m, 0.99)].var for m in METHOD_OPTIONS.values()]
    if min(vals) > 0 and max(vals) / min(vals) > 1.4:
        out.append(("blue", "Model", "The three VaR methods differ by more than 40% at 99% "
                                     "(see Risk tab). Worth knowing which one you're looking at."))

    for w in sanity_warnings(pf, rates):
        out.append(("orange", "Trade data", w))
    return out


# --- overview ---

with tabs[0]:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("##### Things to look at")
        alerts = build_alerts() or [("green", "All clear", "Nothing flagged.")]
        rows = "".join(
            f'<div class="alert"><span class="tag" style="color:{STATUS[c]};'
            f'background:{STATUS[c]}1a">{html.escape(tag)}</span>'
            f'<span>{html_text(msg)}</span></div>' for c, tag, msg in alerts)
        st.markdown(f'<div class="alerts">{rows}</div>', unsafe_allow_html=True)

        st.markdown("##### Risk snapshot")
        top_ccy = risk.currencies["pct_of_var"].idxmax() if risk.var else "-"
        m = st.columns(3)
        m[0].metric("Diversification", fmt_usd(risk.diversification), border=True)
        m[1].metric("ES / VaR", f"{risk.es / risk.var:.2f}x" if risk.var else "-", border=True)
        m[2].metric("VaR / gross",
                    f"{risk.var / pos['usd_notional'].abs().sum() * 1e4:.0f}bp"
                    if risk.var else "-", border=True)
        m = st.columns(3)
        m[0].metric("Top risk ccy",
                    f"{top_ccy} · {risk.currencies.loc[top_ccy, 'pct_of_var']:.0%}"
                    if risk.var else "-", border=True)
        m[1].metric("Trades / pairs", f"{len(pf)} / {pf['ccy_pair'].nunique()}", border=True)
        m[2].metric("Last breach", "none" if bt.empty or not bt["exception"].any()
                    else f"{(asof - bt.index[bt['exception']][-1]).days}d ago", border=True)
        st.caption("Diversification is the sum of standalone VaRs minus portfolio VaR. ES / VaR "
                   "is about 1.15x at 99% for normal moves, higher means fat tails. Top risk ccy "
                   "has the biggest share of VaR. Last breach is the most recent backtest day "
                   "that lost more than VaR.")
    with c2:
        st.markdown("##### Today's P&L by currency")
        attr = daily_pnl_attribution(pf, rates)
        fig = go.Figure(go.Waterfall(
            x=[*attr.index, "Total"], y=[*attr.to_numpy(), 0],
            measure=["relative"] * len(attr) + ["total"],
            increasing=dict(marker_color=GAIN), decreasing=dict(marker_color=LOSS),
            totals=dict(marker_color=BLUE), connector=dict(line=dict(color=MUTED, width=1))))
        show_chart(tidy(fig, 330, yaxis_title="USD", showlegend=False))
        st.caption("Exact split: each currency's USD exposure at the previous close times its "
                   "move since. Trades booked since the last close sit in 'New trades'.")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("##### Cumulative P&L")
        vh = value_history(pf, rates)
        vh = vh[vh.index >= pf["trade_date"].min() - pd.Timedelta(days=3)]
        tot = vh.sum(axis=1)
        fig = go.Figure(go.Scatter(x=tot.index, y=tot, mode="lines", fill="tozeroy",
                                   line=dict(color=BLUE, width=2),
                                   fillcolor="rgba(61,109,179,0.10)"))
        show_chart(tidy(fig, 320, yaxis_title="USD", showlegend=False))
    with c2:
        st.markdown("##### Biggest contributors to VaR")
        top = pos["component_var"].abs().sort_values(ascending=False).head(8).index
        t = pos.loc[top].sort_values("component_var")
        show_chart(hbar(pd.Series(t["component_var"].to_numpy(), index=label(t)), 320,
                        text=[f"{p:.0%}" for p in t["pct_of_var"]], risk=True))
        st.caption("Component VaR of the largest contributors, labelled with their share of "
                   "portfolio VaR. Green bars are trades that hedge the rest of the book.")


# --- positions ---

with tabs[1]:
    st.markdown("##### Trade blotter")
    tbl = pos.reset_index()[[
        "trade_id", "ccy_pair", "side", "notional", "trade_date", "entry_price",
        "spot", "move_pct", "pnl_quote", "pnl_daily_usd", "pnl_inception_usd",
        "pct_of_var"]]
    tbl["trade_date"] = tbl["trade_date"].dt.date
    tbl["quote"] = pf["quote"].to_numpy()
    tbl["pnl_quote"] = [f"{v:,.0f} {q}" for v, q in zip(tbl["pnl_quote"], tbl["quote"])]
    tbl = tbl.drop(columns="quote")
    tbl.columns = ["ID", "Pair", "Side", "Notional", "Traded", "Entry", "Spot", "Move",
                   "P&L (quote)", "Daily $", "Inception $", "% of VaR"]
    money = ["Daily $", "Inception $"]
    st.dataframe(
        tbl.style.format({"Notional": "{:,.0f}", "Entry": "{:,.6g}", "Spot": "{:,.6g}",
                          "Move": "{:+.2%}", "% of VaR": "{:.1%}"}
                         | {c: "{:,.0f}" for c in money})
        .map(pnl_color, subset=["Move", "Daily $", "Inception $"]),
        hide_index=True, height=fit(len(tbl)))
    st.download_button("Download positions with risk (CSV)", pos.to_csv().encode(),
                       file_name=f"positions_{asof:%Y%m%d}.csv", mime="text/csv")

    st.markdown("##### Net position by pair")
    net = net_by_pair(pos).reset_index()
    net.columns = ["Pair", "Trades", "Net notional", "Spot", "USD notional", "Daily P&L $",
                   "Inception P&L $", "Breakeven"]
    st.dataframe(net.style.format({"Net notional": "{:+,.0f}", "Spot": "{:,.6g}",
                                   "USD notional": "{:+,.0f}", "Daily P&L $": "{:,.0f}",
                                   "Inception P&L $": "{:,.0f}", "Breakeven": "{:,.6g}"},
                                  na_rep="flat")
                 .map(pnl_color, subset=["Daily P&L $", "Inception P&L $"]),
                 hide_index=True, height=fit(len(net)))
    st.caption("Breakeven is the spot level where the net position in the pair is flat on P&L.")

    st.markdown("##### Net currency exposure (USD)")
    ce = risk.currencies["exposure_usd"].sort_values() if len(E.columns) else pd.Series()
    show_chart(hbar(ce, 300))
    st.caption("Crosses are split into their two legs, e.g. long EURGBP is long EUR and "
               "short GBP.")

    st.markdown("##### By strategy")
    sp = strat_risk.positions
    strat = pos.groupby("strategy").agg(
        trades=("ccy_pair", "size"),
        gross=("usd_notional", lambda s: s.abs().sum()),
        daily=("pnl_daily_usd", "sum"),
        inception=("pnl_inception_usd", "sum"),
    ).join(sp[["standalone_var", "component_var", "pct_of_var", "incremental_var"]])
    strat["ret_on_risk"] = strat["inception"] / strat["standalone_var"].replace(0, np.nan)
    strat = strat.sort_values("component_var", ascending=False).rename_axis("Strategy")
    strat.columns = ["Trades", "Gross $", "Daily $", "Inception $", "Standalone VaR",
                     "Component VaR", "% of VaR", "Incremental", "P&L / VaR"]
    st.dataframe(strat.style.format({c: "{:,.0f}" for c in strat.columns[1:6]}
                                    | {"Incremental": "{:,.0f}", "% of VaR": "{:.1%}",
                                       "P&L / VaR": "{:+.2f}"}, na_rep="-")
                 .map(pnl_color, subset=["Daily $", "Inception $", "P&L / VaR"]),
                 height=fit(len(strat)))
    st.caption("Strategy VaR is computed on each strategy's netted exposures, so offsetting "
               "trades inside a strategy net off before the decomposition. P&L / VaR is "
               "inception P&L per unit of standalone risk.")


# --- risk ---

with tabs[2]:
    if risk.var == 0:
        st.info("The book has no net currency exposure, so there is no risk to show.")
    n_scen = lookback
    st.markdown(f"**{risk.method}** · {alpha:.0%} · {horizon}-day · {n_scen} daily scenarios "
                f"({returns.index[-n_scen]:%d %b %Y} to {returns.index[-1]:%d %b %Y})")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("##### Standalone vs component VaR by trade")
        rp = pos.sort_values("component_var")
        fig = go.Figure()
        fig.add_bar(x=rp["standalone_var"], y=label(rp), orientation="h", name="Standalone",
                    marker_color=BLUE_PALE)
        fig.add_bar(x=rp["component_var"], y=label(rp), orientation="h", name="Component",
                    marker_color=[LOSS if v > 0 else GAIN for v in rp["component_var"]])
        show_chart(tidy(fig, 440, barmode="overlay", xaxis_title="USD"))
        st.caption("Negative component VaR (green) means the trade hedges the rest of the book. "
                   "The gap between the bars is what diversification saves.")
    with c2:
        st.markdown("##### Component VaR by currency")
        cc = risk.currencies.sort_values("component_var")
        fig = go.Figure(go.Bar(x=cc["component_var"], y=cc.index, orientation="h",
                               marker_color=[LOSS if v > 0 else GAIN for v in cc["component_var"]],
                               text=[f"{p:.0%}" for p in cc["pct_of_var"]], textposition="auto",
                               textangle=0, insidetextfont=dict(color="white")))
        show_chart(tidy(fig, 440, xaxis_title="USD", showlegend=False))

    st.markdown("##### Risk decomposition")
    dec = pos.reset_index()[["trade_id", "ccy_pair", "side", "usd_notional",
                             "standalone_var", "component_var", "pct_of_var",
                             "marginal_var_per_1m", "incremental_var", "component_es"]]
    dec.columns = ["ID", "Pair", "Side", "USD notional", "Standalone", "Component", "% of VaR",
                   "Marginal /+1m", "Incremental", "Component ES"]
    st.dataframe(dec.style.format({"USD notional": "{:+,.0f}", "Standalone": "{:,.0f}",
                                   "Component": "{:,.0f}", "% of VaR": "{:.1%}",
                                   "Marginal /+1m": "{:+,.0f}", "Incremental": "{:,.0f}",
                                   "Component ES": "{:,.0f}"}),
                 hide_index=True, height=fit(len(dec)))
    # totals go in the caption rather than a TOTAL row, marginal and incremental VaR
    # don't have a meaningful total and Streamlit shows blank cells as "None"
    st.caption(md_text(
        f"Totals: standalone VaRs add up to {fmt_usd(pos['standalone_var'].sum())}, "
        f"components add up to portfolio VaR of {fmt_usd(risk.var)} and component ES to "
        f"{fmt_usd(risk.es)}. The difference, {fmt_usd(risk.diversification)}, is the "
        "diversification benefit."))
    with st.expander("How to read this table"):
        st.markdown(
            "- **Standalone VaR**: VaR of the trade on its own.\n"
            "- **Component VaR**: the trade's share of portfolio VaR (Euler allocation). "
            "Components add up to portfolio VaR.\n"
            "- **Marginal /+1m**: change in portfolio VaR from *buying* another 1m of "
            "the base currency. Negative means buying more would reduce risk.\n"
            "- **Incremental VaR**: portfolio VaR minus VaR with the trade closed. Fully "
            "recomputed, so it answers 'what if I close this?' exactly.\n"
            "- **Component ES**: the trade's share of expected shortfall.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### 1-day P&L distribution of today's book")
        dist_key = "hist" if method == "param" else method
        dist = scenario_pnl(E, scenarios_for(dist_key, returns[E.columns], lookback)).sum(axis=1) \
            if len(E.columns) else pd.Series([0.0])
        q = np.quantile(dist, 1 - alpha)
        es1 = -dist[dist <= q].mean()
        fig = go.Figure(go.Histogram(x=dist, nbinsx=60, histnorm="probability density",
                                     marker_color=BLUE_LIGHT, name="Scenarios"))
        sd = np.sqrt(x_net @ ewma_cov(returns[E.columns].iloc[-lookback:]) @ x_net) \
            if len(E.columns) else 0
        if sd > 0:
            xs = np.linspace(dist.min(), dist.max(), 200)
            fig.add_scatter(x=xs, y=stats.norm.pdf(xs, 0, sd), mode="lines", name="Normal (EWMA)",
                            line=dict(color="#5b6474", width=1.5, dash="dot"))
        fig.add_vline(x=q, line_color=LOSS, line_dash="dash",
                      annotation_text=f"VaR {fmt_usd(-q)}", annotation_position="top left")
        fig.add_vline(x=-es1, line_color=LOSS_DARK, line_dash="dot",
                      annotation_text=f"ES {fmt_usd(es1)}", annotation_position="bottom left")
        show_chart(tidy(fig, 340, xaxis_title="USD", yaxis_showticklabels=False))
        st.caption(f"{'Historical' if dist_key == 'hist' else 'Vol-scaled historical'} "
                   "scenarios with the normal curve implied by the EWMA covariance on top. "
                   "Fat tails show up as bars outside the curve.")
    with c2:
        st.markdown("##### Model comparison")
        rows = []
        for (m, a), r in compare.items():
            rows.append({"Method": [k for k, v in METHOD_OPTIONS.items() if v == m][0],
                         "Confidence": f"{a:.0%}", "VaR": r.var, "ES": r.es})
        st.dataframe(pd.DataFrame(rows).style.format({"VaR": "{:,.0f}", "ES": "{:,.0f}"}),
                     hide_index=True, height=fit(len(rows)))
        if svar_start is not None:
            st.markdown(md_text(
                f"Stressed VaR ({alpha:.0%}, {horizon}d): **{fmt_usd(svar)}**, worst 250-day "
                f"window {svar_start:%d %b %Y} to {svar_end:%d %b %Y}, "
                f"{svar / risk.var:.1f}x current VaR."))
        st.caption("Historical keeps fat tails and real co-moves but is slow to react. EWMA "
                   "parametric is quick but assumes normal moves. Filtered historical sits in "
                   "between. A big gap between them is useful information in itself.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Correlation of currency moves vs USD")
        corr = returns[E.columns].iloc[-lookback:].corr()
        fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index,
                                   zmin=-1, zmax=1,
                                   colorscale=[[0, LOSS], [0.5, "#f4f5f7"], [1, BLUE]],
                                   text=corr.round(2).to_numpy(), texttemplate="%{text}",
                                   colorbar=dict(tickfont=dict(size=11))))
        show_chart(tidy(fig, 380))
    with c2:
        st.markdown("##### Volatility, realised vs EWMA (annualised)")
        r_ = returns[E.columns]
        vol = pd.DataFrame({
            "1M": r_.iloc[-21:].std() * np.sqrt(252),
            "3M": r_.iloc[-63:].std() * np.sqrt(252),
            "Lookback": r_.iloc[-lookback:].std() * np.sqrt(252),
            "EWMA": np.sqrt(np.diag(ewma_cov(r_.iloc[-lookback:])) * 252),
        })
        vol["EWMA / lookback"] = vol["EWMA"] / vol["Lookback"]
        st.dataframe(vol.style.format("{:.1%}", subset=["1M", "3M", "Lookback", "EWMA"])
                     .format("{:.2f}x", subset=["EWMA / lookback"])
                     .map(ratio_color, subset=["EWMA / lookback"]), height=fit(len(vol)))
        st.caption("Vol of each currency's USD value. EWMA well above the lookback average "
                   "means risk is elevated right now and plain historical VaR lags.")

    with st.expander("Methodology in short"):
        st.markdown(
            "- **P&L**: a trade of N base units at entry K is worth N·(USD per base) − "
            "N·K·(USD per quote) in USD. Inception P&L is today's value, daily P&L is the "
            "change since the previous close.\n"
            "- **Risk factors**: daily % moves of each currency against USD. Crosses split "
            "into two legs, so netting across pairs is automatic and exact for spot.\n"
            "- **Historical**: today's exposures times each of the last N days' moves.\n"
            "- **Filtered historical**: same, but each day's moves are scaled by EWMA vol now "
            "/ EWMA vol then (Hull-White).\n"
            "- **Parametric**: z·σ with an EWMA (λ = 0.94) covariance.\n"
            "- **Horizon**: √h scaling of the 1-day numbers.\n"
            "More detail and the caveats are in the README.")


# --- hedging ---

with tabs[3]:
    st.markdown("##### Best single hedges")
    st.caption("For each currency, the spot trade that minimises the variance of the whole "
               "book, then VaR recomputed with the hedge on using the selected method. "
               "Currencies you don't hold are included, so this also finds proxy hedges.")
    hedges = best_hedges(E, returns, rates.iloc[-1], method, alpha, horizon, lookback)
    if hedges.empty:
        st.info("Nothing to hedge.")
    else:
        ht = hedges.copy()
        ht["trade"] = ht["side"] + " " + (ht["notional"] / 1e6).map("{:,.1f}m".format) + " " + \
            ht["pair"]
        show = ht[["ccy", "trade", "hedge_usd", "var_after", "var_change"]].copy()
        show["reduction"] = ht["pct_reduction"] * 100
        cfg = st.column_config
        st.dataframe(show, hide_index=True, height=fit(len(show)),
                     column_config={
                         "ccy": "Hedge with", "trade": "Trade",
                         "hedge_usd": cfg.NumberColumn("USD exposure", format="%+,.0f"),
                         "var_after": cfg.NumberColumn("VaR after", format="%,.0f"),
                         "var_change": cfg.NumberColumn("VaR change", format="%+,.0f"),
                         "reduction": cfg.ProgressColumn("VaR reduction", format="%.0f%%",
                                                         min_value=0, max_value=100),
                     })

    st.markdown("##### Test a trade")
    source = pick_one("Trade source", ["Suggested hedge", "Custom trade"], key="trade_source",
                      label_visibility="collapsed")
    trade, trade_date, strategy, ticket = None, asof.date(), "What-if", None

    if source == "Suggested hedge":
        if hedges.empty:
            st.info("No hedges to suggest for this book. Use Custom trade instead.")
        else:
            st.caption("Pick one of the hedges above, or type your own trade into the box, "
                       "e.g. BUY 10m EURJPY or SELL 5m USDJPY @ 150.")
            added = [in_book({"ccy_pair": r.pair, "side": r.side, "notional": r.notional,
                              "trade_date": asof}, check_price=False) for r in ht.itertuples()]
            ht["choice"] = (ht["trade"] + " · VaR " + ht["var_after"].map(fmt_usd)
                            + np.where(added, " · in book", ""))
            # hedges added earlier drop out of the list once the book is hedged, keep showing
            # them (greyed out) while they're still in the book
            listed = set(ht["trade"])
            added_before = st.session_state.get("added_hedges", {})
            kept = [f"{lab} · in book" for lab, r in added_before.items()
                    if lab not in listed and in_book(r, check_price=False)]
            # Hedge labels change with the settings (size and VaR depend on the method) and
            # when a hedge gets added. Keep the selection on the same trade if it's still
            # listed, otherwise go back to the best hedge. Trades typed in by hand are kept.
            options = list(ht["choice"]) + kept
            current = st.session_state.get("hedge_pick")
            if current in st.session_state.get("hedge_options", []) and current not in options:
                same = [o for o in options if o.split(" · ")[0] == current.split(" · ")[0]]
                st.session_state["hedge_pick"] = same[0] if same else options[0]
            st.session_state["hedge_options"] = options
            ticket = st.selectbox("Hedge", options, key="hedge_pick", accept_new_options=True)
            try:
                trade = parse_ticket(ticket) if ticket else None
            except PortfolioError as e:
                st.warning(str(e))
            if ticket in set(options):
                strategy = "Hedge (what-if)"
    else:
        st.caption("Fill in the trade. Leave the entry price empty to book at the market close "
                   "on the trade date (today's spot by default).")
        f = st.columns([1, 1, 1, 1.4, 1.4])
        side = f[0].selectbox("Side", ["BUY", "SELL"], key="ct_side")
        base = f[1].selectbox("Base ccy", SUPPORTED_CCYS, index=SUPPORTED_CCYS.index("EUR"),
                              key="ct_base")
        quote = f[2].selectbox("Quote ccy", SUPPORTED_CCYS, index=SUPPORTED_CCYS.index("USD"),
                               key="ct_quote")
        notional_m = f[3].number_input("Notional (m of base ccy)", min_value=0.0, value=10.0,
                                       step=1.0, format="%.2f", key="ct_notional")
        spot_hint = (f"spot {float(pair_price(base + quote, rates.iloc[-1])):.6g}"
                     if base != quote else "")
        price = f[4].number_input("Entry price (optional)", min_value=0.0, value=None,
                                  format="%.6f", placeholder=spot_hint, key="ct_price")
        g = st.columns([1, 1, 3.8])
        trade_date = g[0].date_input("Trade date", value=asof.date(), max_value=asof.date(),
                                     min_value=rates.index[0].date(), key="ct_date")
        strategy = g[1].text_input("Strategy", value="What-if", key="ct_strategy").strip() \
            or "What-if"
        if base == quote:
            st.warning("Base and quote currency have to be different.")
        elif not notional_m > 0:
            st.warning("Notional has to be more than zero.")
        else:
            trade = {"side": side, "notional": notional_m * 1e6, "ccy_pair": base + quote,
                     "entry_price": price}

    if trade:
        taken = set(st.session_state["book_live"]["trade_id"].astype(str))
        n = 1
        while f"H{n:02d}" in taken:
            n += 1
        row = pd.DataFrame([{"trade_id": f"H{n:02d}", "trade_date": trade_date,
                             "ccy_pair": trade["ccy_pair"], "side": trade["side"],
                             "notional": trade["notional"], "entry_price": trade["entry_price"],
                             "strategy": strategy}])
        # no price means the market close on the trade date, same rule as the editor
        row = fill_at_market(row, rates)
        new_row = row.iloc[0].to_dict()
        new_row["trade_date"] = pd.Timestamp(new_row["trade_date"]).date()

        if in_book(new_row, check_price=trade["entry_price"] is not None):
            # already added: a preview here would show the effect of adding it twice
            st.button("Already in the what-if book", type="primary", disabled=True,
                      help="This trade has been added. Change it, or remove it below, to add "
                           "it again.")
        else:
            # preview the effect before it goes in the book
            trade_pf = load_portfolio(row)
            trade_exp = usd_exposures(trade_pf, rates.iloc[-1])
            x_new = x_net.add(trade_exp.sum(axis=0), fill_value=0.0)
            var_new = var_of(x_new, method, returns, alpha, horizon, lookback)
            pnl_now = float(value_positions(trade_pf, rates)["pnl_inception_usd"].iloc[0])
            # five plain cards, no delta lines, so they all line up at the same height
            m = st.columns([1.3, 1, 1, 1, 1])
            m[0].metric("Trade", f"{trade['side']} {trade['notional'] / 1e6:,.1f}m "
                                 f"{trade['ccy_pair']}", border=True)
            m[1].metric("Entry", f"{new_row['entry_price']:.6g}", border=True,
                        help="Your price if you gave one, otherwise the close on the trade "
                             "date.")
            m[2].metric("VaR with the trade", fmt_usd(var_new), border=True)
            m[3].metric("VaR change", fmt_usd(var_new - risk.var, True), border=True,
                        help="Negative means the trade reduces the book's risk.")
            m[4].metric("P&L at today's spot", fmt_usd(pnl_now, True), border=True,
                        help="Zero for a trade booked at today's spot. With your own price or "
                             "an older date it's the mark-to-market P&L straight away.")
            hedge_label = (ticket.split(" · ")[0] if ticket and source == "Suggested hedge"
                           else None)
            st.button("Add to what-if book", on_click=add_trade, args=(new_row, hedge_label),
                      type="primary")

    # trades added on top of the original book, each removable, plus a way out of what-if
    live = st.session_state["book_live"]
    base_ids = set(st.session_state["base_book"]["trade_id"].astype(str))
    added_rows = live[~live["trade_id"].astype(str).isin(base_ids)]
    if what_if:
        st.markdown("##### Your what-if trades")
        if len(added_rows):
            for r in added_rows.itertuples():
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                price = pd.to_numeric(pd.Series([r.entry_price]), errors="coerce").iloc[0]
                c1.markdown(f"**{r.trade_id}** · {r.side} {float(r.notional) / 1e6:,.1f}m "
                            f"{r.ccy_pair}" + (f" @ {price:.6g}" if np.isfinite(price) else "")
                            + f" · {r.strategy}")
                c2.button("Remove", key=f"remove_{r.trade_id}", on_click=remove_trade,
                          args=(str(r.trade_id),), width="stretch")
        else:
            st.caption("No trades added here, but trades in the original book have been "
                       "edited in the What-if tab. Use Exit what-if at the top to undo that.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### What closing each trade does to VaR")
        inc = pos.sort_values("incremental_var")
        show_chart(hbar(pd.Series(-inc["incremental_var"].to_numpy(), index=label(inc)),
                        380, risk=True))
        st.caption("Change in portfolio VaR if the trade were closed (fully recomputed). "
                   "Green means closing it would cut risk, red means the trade is a hedge "
                   "and closing it would add risk.")
    with c2:
        st.markdown("##### Marginal VaR per +1m base")
        mv = pos.sort_values("marginal_var_per_1m", ascending=False)
        show_chart(hbar(pd.Series(mv["marginal_var_per_1m"].to_numpy(), index=label(mv)),
                        380, risk=True))
        st.caption("Change in VaR from buying another 1m of the base currency in each pair "
                   "(selling is the mirror image). Red means buying adds risk, green means "
                   "buying lowers it. Linear, so only good for small changes.")


# --- stress ---

with tabs[4]:
    rows = []
    for name, shocks in STRESS_SCENARIOS.items():
        rows.append(("Hypothetical", name, stress_pnl(E, shocks)))
    for name, (a, b) in HISTORICAL_EPISODES.items():
        moves = episode_moves(rates, a, b)
        if moves is not None:
            rows.append(("Historical replay", name, stress_pnl(E, moves)))
    ww_pnl, ww_start, ww_end = worst_window(E, returns, 5)
    if ww_start is not None:
        moves = rates.loc[ww_end, ccys] / rates.iloc[rates.index.get_loc(ww_start) - 1][ccys] - 1
        rows.append(("Historical replay",
                     f"Worst 5 days for this book ({ww_start:%d %b} - {ww_end:%d %b %Y})",
                     stress_pnl(E, moves)))

    sc = pd.DataFrame([{
        "Type": t, "Scenario": n, "Portfolio P&L": p.sum(),
        "x VaR": p.sum() / risk.var if risk.var else np.nan,
        "Worst trade": f"{p.idxmin()} {pf.loc[p.idxmin(), 'ccy_pair']}" if len(p) else "",
        "Worst trade P&L": p.min() if len(p) else 0.0,
    } for t, n, p in rows]).sort_values("Portfolio P&L")

    st.markdown("##### Scenario P&L")
    fig = hbar(sc.set_index("Scenario")["Portfolio P&L"].iloc[::-1], 420)
    # both labels sit at the bottom, by the axis, on opposite sides of their lines. The
    # bottom rows are the gain scenarios, so there's nothing to the left of zero there.
    label_bg = "rgba(255,255,255,0.85)"
    fig.add_vline(x=-risk.var, line_color=INK, line_width=1.5, line_dash="dash",
                  annotation_text="-VaR", annotation_position="bottom right",
                  annotation_bgcolor=label_bg)
    if svar_start is not None:
        fig.add_vline(x=-svar, line_color=MUTED, line_width=1.5, line_dash="dot",
                      annotation_text="-stressed VaR", annotation_position="bottom left",
                      annotation_bgcolor=label_bg)
    show_chart(fig)
    st.dataframe(sc.style.format({"Portfolio P&L": "{:,.0f}", "x VaR": "{:+.1f}x",
                                  "Worst trade P&L": "{:,.0f}"})
                 .map(pnl_color, subset=["Portfolio P&L", "Worst trade P&L"]),
                 hide_index=True, height=fit(len(sc)))
    st.caption("Shocks are % moves in each currency against USD and spot trades are revalued "
               "exactly. Historical replays apply the actual close-to-close moves from those "
               "dates to today's book. 'x VaR' compares the result with current VaR.")

    st.markdown("##### Build your own shock")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.caption("% move vs USD, + means the currency strengthens.")
        custom = {}
        cols = st.columns(3)
        for i, c in enumerate(E.columns):
            custom[c] = cols[i % 3].number_input(c, value=0.0, step=1.0, format="%.1f",
                                                 key=f"shock_{c}") / 100
        usd_all = st.slider("Plus a broad USD move (%)", -10.0, 10.0, 0.0, 0.5,
                            help="Positive means USD strengthens against everything.")
        shocks = {c: (1 + v) / (1 + usd_all / 100) - 1 for c, v in custom.items()}
        cp = stress_pnl(E, shocks)
        st.metric("Scenario P&L", fmt_usd(cp.sum(), True),
                  delta=f"{cp.sum() / risk.var:+.1f}x VaR" if risk.var and cp.sum() else None,
                  delta_color="off")
    with c2:
        if any(v != 0 for v in shocks.values()):
            d = cp.sort_values()
            show_chart(hbar(pd.Series(d.to_numpy(), index=label(pf.loc[d.index])), 400,
                            title="P&L by trade"))
        else:
            st.info("Enter some shocks on the left to see P&L by trade.")

    st.markdown("##### Worst days in the lookback for today's book")
    worst = hist_book_pnl.nsmallest(10)
    drivers = []
    for d in worst.index:
        contrib = returns.loc[d, E.columns] * x_net
        top2 = contrib.nsmallest(2).index
        drivers.append(", ".join(f"{c} {returns.loc[d, c]:+.2%}" for c in top2))
    wt = pd.DataFrame({"Date": worst.index.date, "Portfolio P&L": worst.to_numpy(),
                       "x VaR": worst.to_numpy() / var_1d if var_1d else np.nan,
                       "Main drivers (move vs USD)": drivers})
    st.dataframe(wt.style.format({"Portfolio P&L": "{:,.0f}", "x VaR": "{:+.1f}x"})
                 .map(pnl_color, subset=["Portfolio P&L"]), hide_index=True,
                 height=fit(len(wt)))


# --- P&L history and backtest ---

with tabs[5]:
    vh = value_history(pf, rates)
    vh = vh[vh.index >= pf["trade_date"].min() - pd.Timedelta(days=3)]
    by_strat = vh.T.groupby(pf["strategy"]).sum().T
    total = vh.sum(axis=1)
    daily = total.diff().dropna()

    st.markdown("##### Cumulative P&L by strategy (actual trade dates)")
    # The whole book is a light shaded area with its edge drawn green while the book is up
    # since the first trade and red while it's down. The strategy lines sit on top, so
    # the book gives context without a heavy line fighting them for attention.
    fig = go.Figure()
    up, down = split_at_zero(total)
    for part, name, line, fill in [(up, "Book up", GAIN, "rgba(43,154,115,0.14)"),
                                   (down, "Book down", LOSS, "rgba(212,96,79,0.14)")]:
        fig.add_scatter(x=part.index, y=part, name=name, mode="lines", fill="tozeroy",
                        line=dict(width=1.4, color=line), fillcolor=fill,
                        connectgaps=False, hoverinfo="skip")
    # one hover-only series for the tooltip, so it shows the book once per day (the two
    # coloured halves would list it twice near a crossing). Tiny markers coloured by sign
    # under the line: invisible on the chart, but the tooltip swatch turns green or red.
    fig.add_scatter(x=total.index, y=total, name="Whole book", mode="markers",
                    marker=dict(size=1, color=np.where(total >= 0, GAIN, LOSS)),
                    showlegend=False)
    for i, s in enumerate(by_strat.columns):
        fig.add_scatter(x=by_strat.index, y=by_strat[s], name=s, mode="lines",
                        line=dict(width=1.8, color=SERIES[i % len(SERIES)]))
    fig = tidy(fig, 400, yaxis_title="USD")
    # pin the dates, plotly pads the axis around marker series otherwise
    fig.update_layout(hovermode="x unified", xaxis_range=[total.index[0], total.index[-1]])
    # dollar format in the tooltip, leaving out the layers that are drawn but not hovered
    fig.for_each_trace(lambda t: t.update(hovertemplate="%{y:$,.0f}")
                       if t.hoverinfo != "skip" else None)
    show_chart(fig)
    st.caption("The shaded line is the whole book: green while it's up since the first trade, "
               "red while it's down. Hover for every strategy on a given day, click a name in "
               "the legend to hide it or double-click to show it on its own.")

    dd = total - total.cummax()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    m = st.columns(6)
    m[0].metric("Best day", fmt_usd(daily.max(), True), border=True)
    m[1].metric("Worst day", fmt_usd(daily.min(), True), border=True)
    m[2].metric("Up days", f"{(daily > 0).mean():.0%}", border=True)
    m[3].metric("P&L Sharpe", f"{sharpe:.2f}", border=True,
                help="Mean daily P&L / std of daily P&L × √252. The book's size changes as "
                     "trades are added, so read it as a rough guide.")
    m[4].metric("Max drawdown", fmt_usd(dd.min()), border=True)
    m[5].metric("Drawdown now", fmt_usd(dd.iloc[-1]), border=True)

    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", mode="lines",
                               line=dict(color=LOSS, width=1.5),
                               fillcolor="rgba(212,96,79,0.12)"))
    show_chart(tidy(fig, 180, yaxis_title="Drawdown", showlegend=False))

    st.divider()
    st.markdown(f"##### VaR backtest: today's book, 1-day {alpha:.0%}, last {len(bt)} days")
    if bt.empty:
        st.info("Not enough history for a backtest with this lookback.")
    else:
        n, x = len(bt), int(bt["exception"].sum())
        p_kup = kupiec_pof(x, n, alpha)
        _, p_ind = christoffersen(bt["exception"])
        zone = basel_zone(x, n, alpha)
        m = st.columns(5)
        m[0].metric("Exceptions", f"{x} / {n}", border=True)
        m[1].metric("Expected", f"{n * (1 - alpha):.1f}", border=True)
        m[2].metric("Kupiec p", f"{p_kup:.2f}", border=True,
                    help="Tests the number of exceptions. p < 0.05 rejects the model.")
        m[3].metric("Clustering p", f"{p_ind:.2f}", border=True,
                    help="Christoffersen test: do exceptions bunch together? p < 0.05 means "
                         "they cluster, i.e. the model is slow to react to vol changes.")
        m[4].metric("Basel zone", zone or "n/a", border=True,
                    help="Only defined for 250 days at 99%.")
        fig = go.Figure()
        fig.add_bar(x=bt.index, y=bt["pnl"], name="Hypothetical P&L",
                    marker_color=np.where(bt["exception"], LOSS, BLUE_LIGHT))
        fig.add_scatter(x=bt.index, y=-bt["var"], name="-VaR", mode="lines",
                        line=dict(color=LOSS, width=2))
        show_chart(tidy(fig, 360, yaxis_title="USD", bargap=0))
        st.caption(f"Hypothetical P&L is today's exposures applied to each past day's moves. "
                   f"VaR on each day only uses the {bt_lookback} days before it, so this is "
                   "out of sample. Switch the method in the sidebar to compare models.")


# --- market ---

with tabs[6]:
    pairs = sorted(pf["ccy_pair"].unique())
    st.markdown("##### Pairs in the book")
    snap = []
    for p in pairs:
        s = pair_price(p, rates)
        lr = np.log(s).diff().dropna()
        yr = s.iloc[-252:]
        ytd = s[s.index.year == asof.year]
        sd = lr.iloc[-22:-1].std()
        snap.append({"Pair": p, "Spot": s.iloc[-1], "1D": s.iloc[-1] / s.iloc[-2] - 1,
                     "1D / vol": lr.iloc[-1] / sd if sd > 0 else np.nan,
                     "1W": s.iloc[-1] / s.iloc[-6] - 1, "1M": s.iloc[-1] / s.iloc[-22] - 1,
                     "YTD": s.iloc[-1] / ytd.iloc[0] - 1 if len(ytd) > 1 else np.nan,
                     "1M vol": lr.iloc[-21:].std() * np.sqrt(252),
                     "3M vol": lr.iloc[-63:].std() * np.sqrt(252),
                     "1Y range": (s.iloc[-1] - yr.min()) / (yr.max() - yr.min())})
    st.dataframe(pd.DataFrame(snap).style.format(
        {"Spot": "{:,.6g}", "1D": "{:+.2%}", "1D / vol": "{:+.1f}σ", "1W": "{:+.2%}",
         "1M": "{:+.2%}", "YTD": "{:+.2%}", "1M vol": "{:.1%}", "3M vol": "{:.1%}",
         "1Y range": "{:.0%}"}, na_rep="-")
        .map(pnl_color, subset=["1D", "1W", "1M", "YTD"]),
        hide_index=True, height=fit(len(snap)))
    st.caption("1D / vol is today's move in units of the past month's daily vol. 1Y range is "
               "where spot sits between the 1-year low (0%) and high (100%).")

    c1, c2 = st.columns([4, 1])
    with c1:
        pick = pick_one("Pair", pairs, key="market_pair")
    with c2:
        if "market_window" not in st.session_state:
            st.session_state["market_window"] = "1Y"
        window = pick_one("Window", ["3M", "6M", "1Y", "3Y"], key="market_window")
    days = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 10_000}[window]
    s = pair_price(pick, rates).iloc[-days:]
    fig = go.Figure(go.Scatter(x=s.index, y=s, mode="lines", name=pick,
                               line=dict(color=BLUE, width=1.8)))
    for tid, p in pf[pf["ccy_pair"] == pick].iterrows():
        if p["trade_date"] >= s.index[0]:
            buy = p["side"] == "BUY"
            fig.add_scatter(x=[p["trade_date"]], y=[p["entry_price"]], mode="markers+text",
                            text=[f"{tid} {p['side']}"], textposition="top center",
                            marker=dict(size=11, symbol="triangle-up" if buy else "triangle-down",
                                        color=GAIN if buy else LOSS))
    be = net_by_pair(pos).loc[pick, "breakeven"]
    if np.isfinite(be):
        fig.add_hline(y=be, line_dash="dot", line_color=MUTED,
                      annotation_text=f"net breakeven {be:.5g}",
                      annotation_position="bottom left")
    show_chart(tidy(fig, 400, showlegend=False))


# --- what-if extras (bottom of the last tab) ---

with tabs[7]:
    with st.expander("Current spot for the USD pairs"):
        st.dataframe(pd.DataFrame(
            [(usd_pair(c), float(pair_price(usd_pair(c), rates.iloc[-1]))) for c in ccys],
            columns=["Pair", "Spot"]).style.format({"Spot": "{:,.5f}"}), hide_index=True,
            height=fit(len(ccys)))


# A small page script for the few things Streamlit can't do from Python:
#   - a normal scroll over a chart scrolls the page, only a pinch zooms the chart
#   - clicking an open dropdown (or its arrow) closes it, main area dropdowns don't by default
#   - keeps the tab bar pinned right under the header as the header changes height
#   - each tab remembers its own scroll position: a new tab opens at the top, a tab you've
#     been on before opens where you left it
#   - greys out hedges that are already in the what-if book
#   - gives the "Whole book" tooltip entry a proper line swatch
# It runs in a zero-height component and works on the parent page. The guard name has a
# version in it, bump it after changing the script so an open tab picks up the new one.
components.html("""
<script>
const doc = window.parent.document;
if (!doc.__fxdashScriptsV6) {
  doc.__fxdashScriptsV6 = true;
  // a pinch arrives as a wheel event with ctrlKey set, anything else over a chart is
  // stopped before plotly sees it so it scrolls the page instead
  doc.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && e.target.closest && e.target.closest(".js-plotly-plot")) {
      e.stopPropagation();
    }
  }, true);

  // a click on an open dropdown closes it, the same way Escape does
  doc.addEventListener("mousedown", (e) => {
    const box = e.target.closest('[data-baseweb="select"]');
    const input = box && box.querySelector("input");
    if (!input || input.getAttribute("aria-expanded") !== "true") return;
    e.preventDefault();
    e.stopPropagation();
    input.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", code: "Escape",
                                                      keyCode: 27, bubbles: true}));
    input.blur();
  }, true);

  // scroll position per tab. Saved as you scroll, put back the moment the selected tab
  // changes (0 for a tab you haven't scrolled yet, so it opens at the top).
  const scrolled = {};
  const activeTab = () => doc.querySelector('[role="tab"][aria-selected="true"]')?.innerText;
  doc.addEventListener("scroll", (e) => {
    if (e.target.getAttribute && e.target.getAttribute("data-testid") === "stMain") {
      scrolled[activeTab()] = e.target.scrollTop;
    }
  }, true);
  new MutationObserver((changes) => {
    if (!changes.some((c) => c.target.getAttribute("aria-selected") === "true")) return;
    const main = doc.querySelector('[data-testid="stMain"]');
    if (main) main.scrollTop = scrolled[activeTab()] || 0;
  }).observe(doc.body, {attributes: true, subtree: true, attributeFilter: ["aria-selected"]});

  // keep --fx-top-h equal to the pinned header's height, the tab bar is pinned below it
  const sizeHeader = new ResizeObserver((entries) => {
    for (const e of entries) {
      doc.documentElement.style.setProperty("--fx-top-h", e.target.offsetHeight + "px");
    }
  });
  const watchHeader = () => {
    const header = doc.querySelector('[data-testid="stLayoutWrapper"]:has(> .st-key-top_bar)');
    if (header && header !== doc.__fxHeader) {
      if (doc.__fxHeader) sizeHeader.unobserve(doc.__fxHeader);
      doc.__fxHeader = header;
      sizeHeader.observe(header);
    }
  };
  watchHeader();

  // dropdown options for trades already in the what-if book can't be picked again
  new MutationObserver(() => {
    watchHeader();
    doc.querySelectorAll('[role="option"]').forEach((o) => {
      const added = o.innerText.includes("· in book");
      o.style.opacity = added ? "0.45" : "";
      o.style.pointerEvents = added ? "none" : "";
      o.setAttribute("aria-disabled", added ? "true" : "false");
    });
    // the "Whole book" tooltip entry comes from 1px markers, so its swatch is a tiny dot.
    // Redraw it as the same short line the others have, keeping its green or red. Plotly
    // reuses these elements between hovers, so this runs every time and only changes
    // what's different.
    doc.querySelectorAll(".hoverlayer .legend .traces").forEach((t) => {
      const sw = t.querySelector(".legendpoints path");
      if (!sw || !t.textContent.startsWith("Whole book")) return;
      const fill = sw.style.fill;
      const colour = fill && fill !== "none" ? fill : sw.style.stroke;
      if (sw.getAttribute("d") !== "M5,0h30") sw.setAttribute("d", "M5,0h30");
      if (sw.hasAttribute("transform")) sw.removeAttribute("transform");
      if (sw.style.fill !== "none") sw.style.fill = "none";
      if (sw.style.stroke !== colour) sw.style.stroke = colour;
      if (sw.style.strokeWidth !== "1.8px") sw.style.strokeWidth = "1.8px";
    });
  }).observe(doc.body, {childList: true, subtree: true});
}
</script>
""", height=0)
