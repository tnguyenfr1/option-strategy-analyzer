"""
Option Strategy Analyzer — OptionCharts.io / OptionStrat style
==============================================================

A multi-leg option strategy visualizer with:
  • Filled green/red P&L curve at any user-chosen DTE (the "analyze date")
    plus a dashed P&L-at-expiration curve — matching the OptionCharts.io
    "Build a strategy" payoff diagram.
  • A DTE slider so you can scrub the analyze-date curve day-by-day from
    today out to the earliest leg's expiration.
  • A lognormal probability-of-price distribution overlaid on the chart
    so you can see "where is the spot likely to be by my analyze date" —
    matching OptionStrat's premium probability overlay.
  • A "What-If" / Roll-and-Assign simulator: pick any price and any DTE
    and see (a) close-the-position-now P&L, (b) hold-to-expiration P&L,
    (c) assignment outcome if a short leg is ITM at its expiration, and
    (d) the credit or debit of rolling a short leg to a new strike or
    expiration.
  • Per-leg expiration-date dropdown sourced from yfinance, IV per leg,
    and a "Fetch from option chain" button that snaps each leg to a live
    bid/ask mid + IV.

Run:  streamlit run option_strategy_analyzer.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import date, datetime, timedelta

st.set_page_config(
    page_title="Option Strategy Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "## Option Strategy Analyzer\n"
            "Multi-leg payoff diagrams, probability analysis, roll & assignment "
            "simulation, and option price history — powered by live yfinance data.\n\n"
            "**Created by Thuan** in collaboration with **Claude (Anthropic)**.\n\n"
            "For educational purposes only — not financial advice."
        ),
    },
)

# NumPy 2.0+ moved trapz → trapezoid; tolerate either
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ---- Design tokens (terminal palette — keep in sync with .streamlit/config.toml) ----
C_BG      = "#0f1419"   # page background
C_PANEL   = "#171c26"   # card / panel surface
C_BORDER  = "#262d3d"   # hairline borders
C_TEXT    = "#e6e9ef"   # primary text
C_MUTED   = "#8b93a7"   # secondary text
C_GREEN   = "#00c290"   # profit
C_RED     = "#f6465d"   # loss
C_BLUE    = "#4f8cff"   # data / underlying
C_AMBER   = "#f0b90b"   # warnings / strike markers
C_TEAL    = "#2dd4bf"   # expiration curve

# ---- Shared Plotly layout (terminal chart chrome) ----
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", size=12, color=C_TEXT),
    hoverlabel=dict(
        bgcolor=C_PANEL, bordercolor=C_BORDER,
        font=dict(family="IBM Plex Mono, monospace", size=12, color=C_TEXT),
    ),
)
GRID_STYLE = dict(gridcolor="rgba(38,45,61,0.55)", zerolinecolor=C_BORDER,
                  linecolor=C_BORDER)

# ====================================================================
# BLACK-SCHOLES PRICING & GREEKS  (vectorized over S)
# ====================================================================

def bs_price(S, K, T_days, sigma, r, opt_type='call'):
    """Black-Scholes value of a single contract. S may be scalar or array."""
    S = np.asarray(S, dtype=np.float64)
    if T_days <= 0:
        if opt_type == 'call':
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)
    T = T_days / 365.0
    sigma = max(float(sigma), 1e-6)
    sqrtT = np.sqrt(T)
    S_safe = np.where(S > 0, S, 1e-9)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if opt_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T_days, sigma, r, opt_type='call'):
    """Returns (delta, gamma, theta_per_day, vega_per_1pct_IV) at one S."""
    if T_days <= 0:
        if opt_type == 'call':
            d = 1.0 if S > K else 0.0
        else:
            d = -1.0 if S < K else 0.0
        return d, 0.0, 0.0, 0.0
    T = T_days / 365.0
    sigma = max(float(sigma), 1e-6)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if opt_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrtT)
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrtT)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    vega = S * norm.pdf(d1) * sqrtT
    return delta, gamma, theta / 365.0, vega / 100.0


def implied_vol_from_price(price, S, K, T_days, r, opt_type='call'):
    """Invert Black-Scholes for IV given a market price.
    Returns the IV (as decimal, e.g. 0.97 = 97%) or None if no solution.

    This matches what OptionCharts/OptionStrat/most brokers report — they
    compute IV from the live bid/ask mid (or last) rather than trusting
    the data feed's own IV field (which yfinance derives at unspecified
    reference price and is often stale).
    """
    from scipy.optimize import brentq
    if price <= 0 or T_days <= 0 or S <= 0 or K <= 0:
        return None
    # Intrinsic value check: option must be worth at least intrinsic
    if opt_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T_days / 365.0), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T_days / 365.0) - S, 0.0)
    if price < intrinsic - 1e-6:
        return None  # arbitrage / bad data

    def f(sigma):
        return float(bs_price(S, K, T_days, sigma, r, opt_type)) - price

    try:
        # Bracket [0.5%, 500% IV] covers virtually all real-world cases.
        # If f(low) and f(high) have same sign, brentq fails — fall back.
        f_low, f_high = f(0.005), f(5.0)
        if f_low * f_high > 0:
            return None
        return float(brentq(f, 0.005, 5.0, xtol=1e-5, maxiter=100))
    except (ValueError, RuntimeError):
        return None


# ====================================================================
# LEG / POSITION P&L  (handles per-leg expirations, stock legs)
# ====================================================================

def leg_value(leg, S_array, days_elapsed, iv_shift=0.0, r=0.045):
    """Per-contract option value (or per-share stock value) at days_elapsed."""
    if leg['type'] == 'stock':
        return np.asarray(S_array, dtype=np.float64)
    remaining_dte = max(0, leg['dte'] - days_elapsed)
    iv = max(leg['iv'] + iv_shift, 0.01)
    return bs_price(S_array, leg['strike'], remaining_dte, iv, r, leg['type'])


def leg_pnl(leg, S_array, days_elapsed, iv_shift=0.0, r=0.045):
    """Total $ P&L for one leg (qty × 100 for options, qty × 1 for stock)."""
    sign = 1 if leg['action'] == 'buy' else -1
    if leg['type'] == 'stock':
        return sign * leg['qty'] * (np.asarray(S_array, dtype=np.float64) - leg['entry_price'])
    value = leg_value(leg, S_array, days_elapsed, iv_shift, r)
    return sign * leg['qty'] * 100 * (value - leg['entry_price'])


def position_pnl(legs, S_array, days_elapsed, iv_shift=0.0, r=0.045):
    """Sum of P&L across all legs."""
    S = np.asarray(S_array, dtype=np.float64)
    total = np.zeros_like(S)
    for leg in legs:
        total = total + leg_pnl(leg, S, days_elapsed, iv_shift, r)
    return total


def position_cost(legs):
    """Net debit (+) or credit (−) on open."""
    c = 0.0
    for leg in legs:
        sign = 1 if leg['action'] == 'buy' else -1
        mult = 1 if leg['type'] == 'stock' else 100
        c += sign * leg['qty'] * leg['entry_price'] * mult
    return c


# ====================================================================
# PROBABILITY (risk-neutral lognormal)
# ====================================================================

def lognormal_pdf(S, spot, sigma, T_years, r=0.045):
    """Lognormal pdf of S_T given S_0 = spot, vol = sigma, time = T_years."""
    S = np.asarray(S, dtype=np.float64)
    if T_years <= 0 or sigma <= 0:
        return np.zeros_like(S)
    sqrtT_sigma = sigma * np.sqrt(T_years)
    mean_log = np.log(spot) + (r - 0.5 * sigma ** 2) * T_years
    log_S = np.log(np.maximum(S, 1e-9))
    pdf = (1.0 / (S * sqrtT_sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((log_S - mean_log) / sqrtT_sigma) ** 2
    )
    return pdf


def lognormal_cdf(S, spot, sigma, T_years, r=0.045):
    """P(S_T <= S)."""
    if T_years <= 0 or sigma <= 0:
        return np.where(np.asarray(S) >= spot, 1.0, 0.0)
    sqrtT_sigma = sigma * np.sqrt(T_years)
    mean_log = np.log(spot) + (r - 0.5 * sigma ** 2) * T_years
    return norm.cdf((np.log(np.maximum(S, 1e-9)) - mean_log) / sqrtT_sigma)


def prob_in_range(S_lo, S_hi, spot, sigma, T_years, r=0.045):
    """P(S_lo < S_T < S_hi)."""
    if T_years <= 0:
        return 0.0
    return float(lognormal_cdf(S_hi, spot, sigma, T_years, r) -
                 lognormal_cdf(S_lo, spot, sigma, T_years, r))


def implied_move(spot, sigma, T_years):
    """1-sigma expected dollar move (≈ spot × sigma × sqrt(T))."""
    if T_years <= 0 or sigma <= 0:
        return 0.0
    return spot * sigma * np.sqrt(T_years)


# ====================================================================
# STRATEGY METRICS  (Cost, Max P/L, Breakevens, PoP, EV, CVaR, Greeks)
# ====================================================================

def find_breakevens(S_array, pnl_array):
    """Linear-interp x-crossings of P&L curve."""
    bes = []
    for i in range(len(pnl_array) - 1):
        if pnl_array[i] == 0:
            bes.append(float(S_array[i]))
        elif pnl_array[i] * pnl_array[i + 1] < 0:
            x1, x2 = S_array[i], S_array[i + 1]
            y1, y2 = pnl_array[i], pnl_array[i + 1]
            bes.append(float(x1 - y1 * (x2 - x1) / (y2 - y1)))
    return bes


def compute_metrics(legs, spot, sigma, analyze_days_elapsed, iv_shift=0.0,
                    r=0.045, n_points=2000):
    """All strategy metrics. Returns dict, or None if no option legs."""
    option_legs = [l for l in legs if l['type'] != 'stock']
    if not option_legs:
        return None

    min_dte = min(l['dte'] for l in option_legs)
    max_dte = max(l['dte'] for l in option_legs)

    # Wide price grid for probability integration
    S_min = max(0.01, spot * 0.2)
    S_max = spot * 4.0
    S_array = np.linspace(S_min, S_max, n_points)

    # P&L at earliest expiration (when earliest leg becomes intrinsic;
    # later legs revalued by BS with remaining time)
    pnl_at_exp = position_pnl(legs, S_array, days_elapsed=min_dte,
                              iv_shift=iv_shift, r=r)
    # P&L at the user-chosen analyze date
    pnl_at_analyze = position_pnl(legs, S_array, days_elapsed=analyze_days_elapsed,
                                  iv_shift=iv_shift, r=r)

    cost = position_cost(legs)

    max_profit = float(pnl_at_exp.max())
    max_loss = float(pnl_at_exp.min())
    max_profit_price = float(S_array[pnl_at_exp.argmax()])
    max_loss_price = float(S_array[pnl_at_exp.argmin()])

    breakevens = find_breakevens(S_array, pnl_at_exp)
    # Also compute breakevens for the analyze-date curve (what user actually sees
    # on the solid filled line). This matches OptionCharts' chart breakeven label.
    breakevens_analyze = find_breakevens(S_array, pnl_at_analyze)

    # Probability of profit at earliest expiration
    T_min = min_dte / 365.0
    pdf = lognormal_pdf(S_array, spot, sigma, T_min, r)
    pdf_norm = pdf / _trapz(pdf, S_array) if _trapz(pdf, S_array) > 0 else pdf

    # P(P&L > 0) — integrate pdf over profitable price regions
    profitable = pnl_at_exp > 0
    pop = 0.0
    in_seg = False
    seg_start = None
    for i in range(len(S_array)):
        if profitable[i] and not in_seg:
            seg_start = S_array[i]
            in_seg = True
        elif not profitable[i] and in_seg:
            pop += prob_in_range(seg_start, S_array[i - 1], spot, sigma, T_min, r)
            in_seg = False
    if in_seg:
        pop += prob_in_range(seg_start, S_array[-1], spot, sigma, T_min, r)
        if pnl_at_exp[-1] > 0:
            pop += 1 - float(lognormal_cdf(S_array[-1], spot, sigma, T_min, r))

    # Expected value: ∫ pnl × pdf  (over our grid; we lose a bit in the tails)
    expected_value = float(_trapz(pnl_at_exp * pdf_norm, S_array))

    # ---- Probability of MAX PROFIT and MAX LOSS ----
    # Find price regions where P&L is within tolerance of the extreme, then
    # integrate the lognormal pdf over those regions (with tail handling).
    # Tolerance: 0.5% of the P&L range, floor $0.50 — handles flat plateaus
    # (e.g. spreads) without falsely catching the sloped parts.
    pnl_range = max(max_profit - max_loss, 1e-9)
    tol = max(0.005 * pnl_range, 0.50)

    def _prob_region(mask):
        """Integrate pdf probability over contiguous True regions of `mask`,
        extending into the tails if the region touches a grid edge."""
        p = 0.0
        in_seg = False
        seg_start = None
        for j in range(len(S_array)):
            if mask[j] and not in_seg:
                seg_start = S_array[j]
                in_seg = True
            elif not mask[j] and in_seg:
                p += prob_in_range(seg_start, S_array[j - 1], spot, sigma, T_min, r)
                in_seg = False
        if in_seg:
            p += prob_in_range(seg_start, S_array[-1], spot, sigma, T_min, r)
            # Region extends beyond right edge of grid → add right tail
            p += 1 - float(lognormal_cdf(S_array[-1], spot, sigma, T_min, r))
        if mask[0]:
            # Region touches the left edge → add left tail (P below grid start)
            p += float(lognormal_cdf(S_array[0], spot, sigma, T_min, r))
        return min(p, 1.0)

    prob_max_profit = _prob_region(pnl_at_exp >= max_profit - tol)
    prob_max_loss = _prob_region(pnl_at_exp <= max_loss + tol)

    # CVaR(5%) — average P&L in worst 5% of outcomes by probability weight
    dS = S_array[1] - S_array[0]
    weights = pdf_norm * dS
    pairs = sorted(zip(pnl_at_exp, weights), key=lambda x: x[0])
    cum = 0.0
    target = 0.05
    cvar_w_sum = 0.0
    cvar_w = 0.0
    for pnl_val, w in pairs:
        if cum + w <= target:
            cvar_w_sum += pnl_val * w
            cvar_w += w
            cum += w
        else:
            rem = target - cum
            cvar_w_sum += pnl_val * rem
            cvar_w += rem
            break
    cvar = cvar_w_sum / cvar_w if cvar_w > 0 else 0.0

    capital_at_risk = abs(max_loss) if max_loss < 0 else abs(cost)
    expected_return_pct = (expected_value / capital_at_risk * 100) if capital_at_risk > 0 else 0.0
    reward_risk = (max_profit / abs(max_loss)) if abs(max_loss) > 1e-9 else float('inf')

    # Position Greeks at NOW (not at analyze date)
    delta = gamma = theta = vega = 0.0
    for leg in legs:
        sign = 1 if leg['action'] == 'buy' else -1
        if leg['type'] == 'stock':
            delta += sign * leg['qty']
        else:
            d, g, t, v = bs_greeks(spot, leg['strike'], leg['dte'],
                                   leg['iv'] + iv_shift, r, leg['type'])
            mult = sign * leg['qty'] * 100
            delta += d * mult
            gamma += g * mult
            theta += t * mult
            vega += v * mult

    return dict(
        S_array=S_array,
        pnl_at_exp=pnl_at_exp,
        pnl_at_analyze=pnl_at_analyze,
        cost=cost,
        max_profit=max_profit, max_loss=max_loss,
        max_profit_price=max_profit_price, max_loss_price=max_loss_price,
        breakevens=breakevens,
        breakevens_analyze=breakevens_analyze,
        min_dte=min_dte, max_dte=max_dte,
        pop=float(pop),
        prob_max_profit=float(prob_max_profit),
        prob_max_loss=float(prob_max_loss),
        expected_value=expected_value,
        expected_return_pct=expected_return_pct,
        reward_risk=reward_risk,
        cvar=float(cvar),
        capital_at_risk=capital_at_risk,
        delta=delta, gamma=gamma, theta=theta, vega=vega,
    )


# ====================================================================
# YFINANCE HELPERS  (cached)
# ====================================================================

@st.cache_data(ttl=120)
def get_history(symbol, period="365d"):
    return yf.Ticker(symbol).history(period=period)


@st.cache_data(ttl=300)
def get_expirations(symbol):
    """List of expiration date strings ('YYYY-MM-DD') from yfinance."""
    try:
        return list(yf.Ticker(symbol).options) or []
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_chain(symbol, expiration_date):
    """(calls_df, puts_df) or (None, None)."""
    try:
        ch = yf.Ticker(symbol).option_chain(expiration_date)
        return ch.calls, ch.puts
    except Exception:
        return None, None


def dte_from_date_str(exp_str, today=None):
    today = today or datetime.now().date()
    return (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days


def occ_symbol(underlying, exp_str, strike, opt_type):
    """Build the OCC option contract symbol yfinance uses.
    e.g. ('ASST', '2026-07-17', 21.0, 'call') → 'ASST260717C00021000'
    Format: TICKER + YYMMDD + C/P + strike×1000 zero-padded to 8 digits.
    """
    d = datetime.strptime(exp_str, "%Y-%m-%d")
    cp = 'C' if opt_type == 'call' else 'P'
    return f"{underlying.upper()}{d.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


@st.cache_data(ttl=300)
def get_option_history(underlying, exp_str, strike, opt_type, period="1y"):
    """Daily OHLCV history for a specific option contract via yfinance.
    Returns a DataFrame (possibly empty/sparse for illiquid contracts)."""
    try:
        sym = occ_symbol(underlying, exp_str, strike, opt_type)
        hist = yf.Ticker(sym).history(period=period)
        return hist if hist is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def lookup_option_quote(symbol, exp_str, strike, opt_type):
    """Return dict with bid/ask/mid/last/iv for the matching contract."""
    calls, puts = get_chain(symbol, exp_str)
    df = calls if opt_type == 'call' else puts
    if df is None or df.empty:
        return None
    match = df[(df['strike'] - strike).abs() < 0.01]
    if match.empty:
        idx = (df['strike'] - strike).abs().idxmin()
        return dict(found=False, closest_strike=float(df.loc[idx, 'strike']))
    row = match.iloc[0]
    bid = float(row.get('bid') or 0)
    ask = float(row.get('ask') or 0)
    last = float(row.get('lastPrice') or 0)
    iv = float(row.get('impliedVolatility') or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    return dict(
        found=True, bid=bid, ask=ask, mid=mid, last=last, iv=iv,
        volume=int(row.get('volume') or 0),
        open_interest=int(row.get('openInterest') or 0),
    )


def list_strikes(symbol, exp_str, opt_type):
    """Sorted list of strikes for a given expiration / type."""
    calls, puts = get_chain(symbol, exp_str)
    df = calls if opt_type == 'call' else puts
    if df is None or df.empty:
        return []
    return sorted(df['strike'].unique().tolist())


# ====================================================================
# CHART BUILDER  (OptionCharts-style: filled P&L + dashed exp curve
#                 + lognormal prob overlay on hidden 2nd y-axis)
# ====================================================================

def build_pnl_chart(metrics, spot, sigma, analyze_days_elapsed, ticker,
                    view_low, view_high, show_prob=True, r=0.045):
    S = metrics['S_array']
    mask = (S >= view_low) & (S <= view_high)
    S_v = S[mask]
    pnl_exp_v = metrics['pnl_at_exp'][mask]
    pnl_an_v = metrics['pnl_at_analyze'][mask]

    # Split the analyze-date curve into above-zero and below-zero so we can
    # fill green above and red below, matching OptionCharts' look.
    pnl_an_pos = np.where(pnl_an_v >= 0, pnl_an_v, 0)
    pnl_an_neg = np.where(pnl_an_v <= 0, pnl_an_v, 0)

    # Effective DTE for the analyze curve (relative to earliest leg expiry)
    eff_dte_remaining = max(0, metrics['min_dte'] - analyze_days_elapsed)
    analyze_label = (f"P&L @ Expiration"
                     if eff_dte_remaining == 0
                     else f"P&L @ {eff_dte_remaining}d to earliest expiration")

    # Probability overlay
    T_years = max(1, analyze_days_elapsed) / 365.0 if analyze_days_elapsed > 0 else metrics['min_dte'] / 365.0
    # Use the time we're "evaluating at" — analyze date if > 0, else min_dte
    T_for_prob = analyze_days_elapsed / 365.0 if analyze_days_elapsed > 0 else metrics['min_dte'] / 365.0
    pdf_v = lognormal_pdf(S_v, spot, sigma, T_for_prob, r)
    pdf_max = float(pdf_v.max()) if pdf_v.max() > 0 else 1.0
    # P(below) / P(above) for hover
    cdf_v = lognormal_cdf(S_v, spot, sigma, T_for_prob, r)
    prob_above = (1 - cdf_v) * 100
    prob_below = cdf_v * 100

    # Hover text (combine both curves + probability)
    cost = metrics['cost']
    cost_basis_for_pct = abs(cost) if abs(cost) > 1 else 1.0
    hover = [
        f"<b>{ticker}: ${s:.2f}</b> ({(s/spot-1)*100:+.2f}% from spot)<br>"
        f"<b>P&L @ analyze:</b> ${pa:+,.2f} ({pa/cost_basis_for_pct*100:+.1f}%)<br>"
        f"<b>P&L @ expiration:</b> ${pe:+,.2f} ({pe/cost_basis_for_pct*100:+.1f}%)<br>"
        f"<b>Prob below ${s:.2f}:</b> {pb:.1f}%<br>"
        f"<b>Prob above ${s:.2f}:</b> {pa_above:.1f}%"
        for s, pa, pe, pb, pa_above in zip(S_v, pnl_an_v, pnl_exp_v, prob_below, prob_above)
    ]

    fig = go.Figure()

    # --- Probability distribution overlay (hidden secondary y-axis) ---
    if show_prob and pdf_max > 0:
        fig.add_trace(go.Scatter(
            x=S_v, y=pdf_v,
            mode='lines',
            line=dict(color='rgba(79,140,255,0.45)', width=1),
            fill='tozeroy',
            fillcolor='rgba(79,140,255,0.10)',
            name=f'Probability density ({int(analyze_days_elapsed)}d)',
            yaxis='y2',
            hoverinfo='skip',
            showlegend=True,
        ))
        # Implied move markers (±1σ, ±2σ)
        im1 = implied_move(spot, sigma, T_for_prob)
        for k, dash, op in [(1, 'dot', 0.40), (2, 'dot', 0.22)]:
            for sign in (-1, 1):
                x = spot + sign * k * im1
                if view_low <= x <= view_high:
                    fig.add_vline(
                        x=x, line=dict(color=f'rgba(79,140,255,{op})', width=1, dash=dash),
                    )

    # --- P&L @ expiration: dashed thin line, no fill (matches screenshot) ---
    fig.add_trace(go.Scatter(
        x=S_v, y=pnl_exp_v,
        mode='lines',
        line=dict(color='rgba(45,212,191,0.75)', width=1.5, dash='dash'),
        name='P&L @ Expiration',
        hoverinfo='skip',
    ))

    # --- P&L @ analyze date: filled, green above zero, red below ---
    # Positive region (green fill)
    fig.add_trace(go.Scatter(
        x=S_v, y=pnl_an_pos,
        mode='lines',
        line=dict(color='#00c290', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,194,144,0.22)',
        name=analyze_label,
        text=hover,
        hovertemplate='%{text}<extra></extra>',
    ))
    # Negative region (red fill) — share legend with positive
    fig.add_trace(go.Scatter(
        x=S_v, y=pnl_an_neg,
        mode='lines',
        line=dict(color='#f6465d', width=2),
        fill='tozeroy',
        fillcolor='rgba(246,70,93,0.18)',
        name='P&L @ analyze (loss)',
        showlegend=False,
        hoverinfo='skip',
    ))

    # Current price line
    fig.add_vline(
        x=spot, line=dict(color=C_BLUE, width=1.2, dash='dash'),
        annotation_text=f"Current: ${spot:.2f}", annotation_position="top",
        annotation_font=dict(color=C_BLUE, size=10,
                             family="IBM Plex Mono, monospace"),
    )
    # Breakeven lines — use analyze-date curve (the SOLID curve the user sees),
    # since that's where the on-chart "Breakeven: $X" label crosses zero.
    chart_bes = metrics.get('breakevens_analyze') or metrics['breakevens']
    for i, be in enumerate(chart_bes):
        if view_low <= be <= view_high:
            fig.add_vline(
                x=be, line=dict(color=C_AMBER, width=1, dash='dot'),
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="top" if i % 2 == 0 else "bottom",
                annotation_font=dict(color=C_AMBER, size=10,
                                     family="IBM Plex Mono, monospace"),
            )
    # Zero line
    fig.add_hline(y=0, line=dict(color='rgba(139,147,167,0.45)', width=1))

    fig.update_layout(
        **PLOTLY_BASE,
        xaxis=dict(
            title=f"{ticker} Price ($)",
            showgrid=True, zeroline=False, range=[view_low, view_high],
            tickprefix="$", **GRID_STYLE,
        ),
        yaxis=dict(
            title="Expected Profit & Loss ($)",
            showgrid=True, zeroline=False, tickformat='+$,.0f',
            **GRID_STYLE,
        ),
        yaxis2=dict(
            title=dict(
                text="Probability density",
                font=dict(color=C_BLUE, size=11),
            ),
            overlaying='y', side='right',
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=C_BLUE, size=10,
                          family="IBM Plex Mono, monospace"),
            tickformat='.3f',
            range=[0, pdf_max * 4.5],  # squash so PDF occupies bottom ~22%
            fixedrange=True,
            zeroline=False,
        ),
        height=480,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=70, t=50, b=60),
    )
    return fig


# ====================================================================
# OPTION PRICE HISTORY CHART  (real candles + theoretical BS + underlying)
# ====================================================================

def build_option_history_chart(underlying_hist, option_hist, leg, ticker,
                               r=0.045, period_days=365):
    """Option price over time vs. the underlying — like OptionCharts' contract
    chart. Plots:
      • Real option OHLC candles where trade history exists (green/red)
      • Theoretical BS price line over the FULL window, computed from the
        underlying's daily closes using the leg's current IV (dashed)
      • Underlying close on a secondary axis (thin blue)
      • Option volume bars along the bottom
    Returns (fig, n_real_bars) or (None, 0) if no underlying history.
    """
    if underlying_hist is None or underlying_hist.empty:
        return None, 0

    uh = underlying_hist.copy()
    uh.index = pd.to_datetime(uh.index).tz_localize(None)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
    uh = uh[uh.index >= cutoff]
    if uh.empty:
        return None, 0

    exp_date = pd.Timestamp(datetime.strptime(leg['expiration'], "%Y-%m-%d"))
    iv = max(float(leg['iv']), 0.01)
    K = float(leg['strike'])
    opt_type = leg['type']

    # Theoretical BS price for every day in the underlying history
    days_to_exp = (exp_date - uh.index).days.values.astype(float)
    theo = np.array([
        float(bs_price(s, K, max(d, 0), iv, r, opt_type))
        for s, d in zip(uh['Close'].values, days_to_exp)
    ])

    fig = go.Figure()

    # --- Theoretical line (full window) ---
    fig.add_trace(go.Scatter(
        x=uh.index, y=theo,
        name=f"Theoretical (BS @ {iv*100:.0f}% IV)",
        line=dict(color=C_MUTED, width=1.5, dash='dot'),
        hovertemplate="%{x|%b %d}: $%{y:.2f}<extra>Theoretical</extra>",
    ))

    # --- Real option candles (where they exist) ---
    n_real = 0
    oh = option_hist
    if oh is not None and not oh.empty:
        oh = oh.copy()
        oh.index = pd.to_datetime(oh.index).tz_localize(None)
        oh = oh[oh.index >= cutoff]
        # Filter out zero-price rows (no trades)
        oh = oh[(oh['Close'] > 0) | (oh['Open'] > 0)]
        n_real = len(oh)
        if n_real > 0:
            fig.add_trace(go.Candlestick(
                x=oh.index,
                open=oh['Open'], high=oh['High'],
                low=oh['Low'], close=oh['Close'],
                name="Option price (real trades)",
                increasing=dict(line=dict(color=C_GREEN, width=1),
                                fillcolor=C_GREEN),
                decreasing=dict(line=dict(color=C_RED, width=1),
                                fillcolor=C_RED),
            ))
            # Volume bars on a third (hidden) axis at the bottom
            if 'Volume' in oh.columns and oh['Volume'].sum() > 0:
                vol_colors = [C_GREEN if c >= o else C_RED
                              for c, o in zip(oh['Close'], oh['Open'])]
                fig.add_trace(go.Bar(
                    x=oh.index, y=oh['Volume'],
                    name="Volume", yaxis='y3',
                    marker_color=vol_colors, opacity=0.45,
                    hovertemplate="%{x|%b %d}: %{y:,.0f}<extra>Volume</extra>",
                ))

    # --- Underlying close (secondary axis) ---
    fig.add_trace(go.Scatter(
        x=uh.index, y=uh['Close'],
        name=f"{ticker} price",
        yaxis='y2',
        line=dict(color=C_BLUE, width=1.2),
        hovertemplate="%{x|%b %d}: $%{y:.2f}<extra>" + ticker + "</extra>",
    ))

    # Strike reference line on the underlying axis
    fig.add_hline(y=K, line=dict(color=C_AMBER, width=1, dash='dash'),
                  yref='y2',
                  annotation_text=f"Strike ${K:.2f}",
                  annotation_font=dict(color=C_AMBER, size=10,
                                       family="IBM Plex Mono, monospace"),
                  annotation_position="top right")

    opt_label = (f"{ticker} {datetime.strptime(leg['expiration'], '%Y-%m-%d').strftime('%b %d, %Y')} "
                 f"${K:.2f} {opt_type.upper()}")
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"{opt_label}", font=dict(
            size=13, family="IBM Plex Mono, monospace", color=C_MUTED)),
        xaxis=dict(title="", rangeslider=dict(visible=False), **GRID_STYLE),
        yaxis=dict(
            title=dict(text="Option price ($)", font=dict(size=11)),
            side='left', tickprefix="$",
            domain=[0.22, 1.0],   # reserve bottom for volume bars
            **GRID_STYLE,
        ),
        yaxis2=dict(
            title=dict(text=f"{ticker} price ($)", font=dict(color=C_BLUE, size=11)),
            overlaying='y', side='right',
            showgrid=False, tickprefix="$",
            tickfont=dict(color=C_BLUE, size=10,
                          family="IBM Plex Mono, monospace"),
        ),
        yaxis3=dict(
            domain=[0.0, 0.18],
            anchor='x',
            showgrid=False,
            tickfont=dict(size=9),
            title=dict(text="Vol", font=dict(size=10)),
        ),
        height=520,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=70, t=60, b=40),
    )
    return fig, n_real


# ====================================================================
# SCENARIO SIMULATOR  (price × DTE table + per-leg breakdown)
# ====================================================================

def evaluate_scenario(legs, S_target, days_elapsed, iv_shift=0.0, r=0.045):
    """Detailed P&L at one (S_target, days_elapsed) — by leg + totals."""
    rows = []
    total_close = 0.0     # P&L if you close all legs at theoretical mid
    total_exp = 0.0       # P&L if you hold each leg to its own expiration
    for i, leg in enumerate(legs):
        sign = 1 if leg['action'] == 'buy' else -1
        mult = 1 if leg['type'] == 'stock' else 100
        if leg['type'] == 'stock':
            val_now = S_target
            val_exp = S_target
            pnl_now = sign * leg['qty'] * (val_now - leg['entry_price'])
            pnl_exp = pnl_now
        else:
            rem = max(0, leg['dte'] - days_elapsed)
            iv = max(leg['iv'] + iv_shift, 0.01)
            val_now = float(bs_price(S_target, leg['strike'], rem, iv, r, leg['type']))
            if leg['type'] == 'call':
                val_exp = max(S_target - leg['strike'], 0.0)
            else:
                val_exp = max(leg['strike'] - S_target, 0.0)
            pnl_now = sign * leg['qty'] * 100 * (val_now - leg['entry_price'])
            pnl_exp = sign * leg['qty'] * 100 * (val_exp - leg['entry_price'])
        total_close += pnl_now
        total_exp += pnl_exp
        rows.append(dict(
            leg=i + 1,
            spec=_leg_label(leg),
            entry=leg['entry_price'],
            value_now=val_now,
            close_pnl=pnl_now,
            exp_pnl=pnl_exp,
            itm_at_exp=_is_itm_at_exp(leg, S_target),
        ))
    return dict(rows=rows, close_pnl=total_close, exp_pnl=total_exp)


def _leg_label(leg):
    sign = "+" if leg['action'] == 'buy' else "-"
    if leg['type'] == 'stock':
        return f"{sign}{leg['qty']} shares @ ${leg['entry_price']:.2f}"
    return (f"{sign}{leg['qty']}× {leg['dte']}d "
            f"{leg['strike']:.2f} {leg['type'].upper()} @ ${leg['entry_price']:.2f}")


def _is_itm_at_exp(leg, S_at_exp):
    if leg['type'] == 'call':
        return S_at_exp > leg['strike']
    if leg['type'] == 'put':
        return S_at_exp < leg['strike']
    return False


# ====================================================================
# ============== STREAMLIT UI ========================================
# ====================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

/* ---------- The signature: every number speaks mono ---------- */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
.mono, code, .stNumberInput input, [data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-variant-numeric: tabular-nums;
}

/* ---------- Metric tiles → bordered terminal cards (responsive) ---------- */
div[data-testid="stMetric"] {
    background: #171c26;
    border: 1px solid #262d3d;
    border-radius: 6px;
    padding: 0.7rem 0.9rem 0.6rem 0.9rem;
    min-width: 0;                       /* allow card to shrink inside flex */
    overflow: hidden;
}
div[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b93a7 !important;
    font-weight: 600;
    white-space: normal !important;     /* let long labels wrap, not clip */
}
[data-testid="stMetricValue"] {
    /* Fluid size: shrinks smoothly as the window narrows */
    font-size: clamp(0.95rem, 0.55rem + 0.9vw, 1.45rem) !important;
    font-weight: 500;
    white-space: normal !important;
    overflow-wrap: break-word;
    line-height: 1.25;
}
[data-testid="stMetricDelta"] {
    font-size: clamp(0.65rem, 0.5rem + 0.3vw, 0.8rem) !important;
    white-space: normal !important;
}

/* Let Streamlit columns shrink below their content width instead of clipping */
div[data-testid="column"] { min-width: 0 !important; }
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }

/* Narrow screens: tighter card padding, smaller eyebrows */
@media (max-width: 1200px) {
    div[data-testid="stMetric"] { padding: 0.5rem 0.6rem 0.45rem 0.6rem; }
    div[data-testid="stMetric"] label { font-size: 0.6rem !important; letter-spacing: 0.05em; }
}
@media (max-width: 800px) {
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    .section-title { font-size: 1.05rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}

/* ---------- Section headers: tick + uppercase eyebrow ---------- */
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #8b93a7;
    margin: 2.2rem 0 0.2rem 0;
}
.section-eyebrow::before { content: "▍"; color: #00c290; margin-right: 6px; }
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e6e9ef;
    margin: 0 0 0.8rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid #262d3d;
}

/* ---------- Column header labels in the leg editor ---------- */
.small-cap {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #8b93a7;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
}

/* ---------- Inputs / selects: flat terminal fields ---------- */
.stNumberInput input, .stTextInput input,
[data-baseweb="select"] > div {
    background-color: #171c26 !important;
    border-color: #262d3d !important;
    border-radius: 5px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #00c290 !important;
    box-shadow: 0 0 0 1px #00c29044 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 5px;
    border: 1px solid #262d3d;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"] {
    background: #00c290; border-color: #00c290; color: #0f1419;
}
.stButton > button[kind="primary"]:hover { background: #00dba2; border-color: #00dba2; }

/* ---------- Tabs: underline style ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid #262d3d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.03em;
    background: transparent; border-radius: 5px 5px 0 0;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #171c26;
    border-bottom: 2px solid #00c290;
}

/* ---------- Expanders & dataframes: panel chrome ---------- */
details[data-testid="stExpander"] {
    background: #171c26;
    border: 1px solid #262d3d !important;
    border-radius: 6px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #262d3d; border-radius: 6px;
}

/* ---------- Sidebar polish ---------- */
[data-testid="stSidebar"] {
    border-right: 1px solid #262d3d;
}
[data-testid="stSidebar"] h2 {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #8b93a7; font-family: 'IBM Plex Mono', monospace;
}

/* ---------- Strategy summary card ---------- */
.strategy-card {
    background: #171c26;
    border: 1px solid #262d3d;
    border-left: 3px solid #00c290;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 1rem;
}
.strategy-card .mono { font-size: 0.9rem; }

/* Trim default top padding so the header sits like a terminal toolbar */
.block-container { padding-top: 2.2rem; }
hr { border-color: #262d3d !important; }
</style>
""", unsafe_allow_html=True)


def section(eyebrow, title):
    """Terminal-style section header: mono eyebrow + bordered title."""
    st.markdown(
        f"<div class='section-eyebrow'>{eyebrow}</div>"
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    "<h1 style='margin-bottom:0.1rem;'>Option Strategy Analyzer</h1>"
    "<p style='color:#8b93a7; font-size:0.88rem; margin-top:0;'>"
    "Multi-leg payoff · probability analysis · roll &amp; assignment simulation "
    "· live chain data</p>",
    unsafe_allow_html=True,
)

# --------------- SIDEBAR --------------------------------------------
st.sidebar.header("Underlying")
ticker = st.sidebar.text_input("Ticker", value="ASST").upper().strip() or "SPY"

try:
    hist = get_history(ticker)
    auto_spot = float(hist['Close'].iloc[-1])
    log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    realized_vol = float(log_ret.std() * np.sqrt(252))
except Exception:
    auto_spot, realized_vol = 100.0, 0.4
    hist = pd.DataFrame()
    st.sidebar.warning(f"Couldn't fetch live data for {ticker}; using defaults.")

spot = st.sidebar.number_input("Spot price ($)", value=auto_spot, step=0.5, format="%.2f")
sigma_default = st.sidebar.number_input(
    f"IV for probability calcs (realized 1y: {realized_vol*100:.0f}%)",
    value=realized_vol * 100, step=1.0,
) / 100
r = st.sidebar.number_input("Risk-free rate (%)", value=4.5, step=0.1) / 100

st.sidebar.markdown("---")
st.sidebar.header("Chart controls")
view_choice = st.sidebar.selectbox(
    "View range",
    ["±20%", "±40%", "±60%", "±100%", "All"],
    index=1,
)
show_prob = st.sidebar.checkbox("Show probability distribution overlay", value=True)
iv_shift_pct = st.sidebar.slider(
    "IV shift across all legs (±%)",
    -50, 50, 0, 1,
    help="Stress test: shift every leg's IV by this many percentage points.",
)
iv_shift = iv_shift_pct / 100

iv_from_mid = st.sidebar.checkbox(
    "Compute IV from chain mid (recommended)",
    value=True,
    help="If on: invert Black-Scholes against the bid/ask mid to derive IV "
         "(matches OptionCharts/OptionStrat methodology).\n\n"
         "If off: use the IV that yfinance returns directly. yfinance's IV "
         "is often stale or computed at the last trade price, which can "
         "differ from the live mid by several percentage points.",
)

# --------------- POSITION EDITOR ------------------------------------
section("Position", "Options")
st.caption("💡 Edit fields directly, or use **Fetch from option chain** to pull live bid/ask mids and IV from yfinance.")

# Default to the ASST diagonal call spread from the user's screenshot
if 'legs' not in st.session_state or st.session_state.get('last_ticker') != ticker:
    st.session_state.last_ticker = ticker
    today = datetime.now().date()
    # Pick two expirations from yfinance if available, else 30 / 60 days
    exps = get_expirations(ticker)
    if len(exps) >= 2:
        # Front month ~30 DTE, back month ~60 DTE
        candidates = [(e, dte_from_date_str(e)) for e in exps if dte_from_date_str(e) >= 7]
        candidates.sort(key=lambda x: x[1])
        front = candidates[0] if candidates else (exps[0], 30)
        back = next(((e, d) for e, d in candidates if d > front[1] + 10), front)
        # Snap to REAL chain strikes (nearest OTM-ish call)
        front_strikes = list_strikes(ticker, front[0], 'call')
        back_strikes = list_strikes(ticker, back[0], 'call')
        front_target = auto_spot * 1.05
        back_target = auto_spot * 1.20
        front_strike = (min(front_strikes, key=lambda s: abs(s - front_target))
                        if front_strikes else round(front_target, 2))
        back_strike = (min(back_strikes, key=lambda s: abs(s - back_target))
                       if back_strikes else round(back_target, 2))
        st.session_state.legs = [
            {'action': 'sell', 'qty': 1, 'type': 'call',
             'strike': back_strike,
             'expiration': back[0], 'dte': back[1],
             'entry_price': 1.50, 'iv': realized_vol},
            {'action': 'buy', 'qty': 1, 'type': 'call',
             'strike': front_strike,
             'expiration': front[0], 'dte': front[1],
             'entry_price': 1.80, 'iv': realized_vol},
        ]
    else:
        st.session_state.legs = [
            {'action': 'sell', 'qty': 1, 'type': 'call',
             'strike': round(auto_spot * 1.20, 2),
             'expiration': '', 'dte': 60,
             'entry_price': 1.50, 'iv': realized_vol},
            {'action': 'buy', 'qty': 1, 'type': 'call',
             'strike': round(auto_spot * 1.05, 2),
             'expiration': '', 'dte': 30,
             'entry_price': 1.80, 'iv': realized_vol},
        ]

# --- Chain-sync controls ---
fcol1, fcol2, fcol3 = st.columns([1.4, 1.4, 4])
force_refetch = fcol1.button("🔄 Re-fetch chain quotes", type="primary", use_container_width=True,
                              help="Force a refresh of all legs against the live chain (overrides any manual edits).")
use_mid = fcol2.checkbox("Use bid/ask mid (else last)", value=True)

# Force-refresh wipes the per-leg sync cache so the auto-sync block below re-pulls everything.
if force_refetch:
    for leg in st.session_state.legs:
        leg.pop('_synced_key', None)
        leg['_manual_edit'] = False
    st.rerun()


def _sync_leg_to_chain(leg, ticker, use_mid, leg_idx=None,
                       spot=None, r=0.045, iv_from_mid=True):
    """Auto-pull entry_price + IV from the chain when (exp, strike, type) changes.
    Mutates `leg` in place AND, when leg_idx is provided, also writes the new
    values into the corresponding Streamlit widget keys so the rendered fields
    refresh on the next paint (widgets with `key=` are owned by session_state).

    When `iv_from_mid=True` and spot is provided, IV is recomputed by inverting
    Black-Scholes against the mid price — this matches OptionCharts/OptionStrat
    and is more accurate than yfinance's own IV field (which is often stale).

    Manual user edits to entry/iv are preserved as long as (exp, strike, type)
    hasn't changed.
    """
    if leg['type'] == 'stock' or not leg.get('expiration'):
        leg['_chain_status'] = 'na'
        return
    sync_key = (leg['expiration'], round(float(leg['strike']), 4), leg['type'])
    if leg.get('_synced_key') == sync_key:
        return  # already synced for this combo — keep any manual edits
    q = lookup_option_quote(ticker, leg['expiration'], leg['strike'], leg['type'])
    if q is None:
        leg['_chain_status'] = 'no_chain'
    elif not q.get('found'):
        leg['_chain_status'] = 'no_strike'
        leg['_closest_strike'] = q.get('closest_strike')
    else:
        price = q['mid'] if use_mid else q['last']
        if price > 0:
            leg['entry_price'] = round(price, 2)
            # Push to widget session state so the displayed number_input updates
            if leg_idx is not None:
                _sk = f"px_{leg_idx}"
                if _sk in st.session_state:
                    st.session_state[_sk] = float(leg['entry_price'])

        # ----- IV: compute from mid via BS inverse (preferred), else yfinance's IV -----
        yf_iv = q.get('iv', 0)
        leg['_chain_iv_yf'] = yf_iv  # remember yfinance's reading for the readout
        computed_iv = None
        if iv_from_mid and spot is not None and price > 0:
            dte = dte_from_date_str(leg['expiration'])
            computed_iv = implied_vol_from_price(
                price, spot, leg['strike'], dte, r, leg['type']
            )
        chosen_iv = computed_iv if computed_iv is not None else yf_iv
        if chosen_iv > 0:
            leg['iv'] = round(chosen_iv, 4)
            if leg_idx is not None:
                _sk = f"iv_{leg_idx}"
                if _sk in st.session_state:
                    # Widget shows percentage, so store *100
                    st.session_state[_sk] = float(leg['iv']) * 100

        leg['_chain_status'] = 'ok' if price > 0 else 'no_price'
        leg['_chain_bid'] = q.get('bid', 0)
        leg['_chain_ask'] = q.get('ask', 0)
        leg['_chain_mid'] = q.get('mid', 0)
        leg['_chain_last'] = q.get('last', 0)
        leg['_chain_iv'] = chosen_iv  # the IV we actually adopted
        leg['_chain_iv_computed'] = computed_iv  # may be None
        leg['_chain_volume'] = q.get('volume', 0)
        leg['_chain_oi'] = q.get('open_interest', 0)
    leg['_synced_key'] = sync_key
    leg['_manual_edit'] = False

# Render leg editor (matches screenshot layout)
hdr = st.columns([0.6, 0.85, 0.5, 2.0, 1.0, 0.85, 1.05, 0.85, 0.9, 0.3])
for col, lbl in zip(hdr, ["", "Action", "Qty", "Expiration", "Strike", "Type",
                          "Entry $", "Δ (calc)", "IV %", ""]):
    col.markdown(f"<span class='small-cap'>{lbl}</span>", unsafe_allow_html=True)

# Get expirations once for all rows
exps_list = get_expirations(ticker)
exp_options = [(e, dte_from_date_str(e)) for e in exps_list]
# Filter to future only and sort
exp_options = [(e, d) for e, d in exp_options if d >= 0]
exp_options.sort(key=lambda x: x[1])

to_remove = None
for i, leg in enumerate(st.session_state.legs):
    cols = st.columns([0.6, 0.85, 0.5, 2.0, 1.0, 0.85, 1.05, 0.85, 0.9, 0.3])
    cols[0].markdown(f"**{ticker}**")
    leg['action'] = cols[1].selectbox(
        "", ['buy', 'sell'],
        index=(0 if leg['action'] == 'buy' else 1),
        key=f"act_{i}", label_visibility="collapsed",
    )
    leg['qty'] = cols[2].number_input(
        "", min_value=1, value=int(leg['qty']), step=1,
        key=f"qty_{i}", label_visibility="collapsed",
    )

    if leg['type'] == 'stock':
        cols[3].markdown("*(stock)*")
        leg['expiration'] = ''
        leg['dte'] = 0
        cols[4].markdown("*(N/A)*")
        leg['strike'] = 0.0
        cols[5].markdown("*(stock)*")
        cols[7].markdown(f"**{1.00 if leg['action']=='buy' else -1.00:+.2f}**")
        cols[8].markdown("*(N/A)*")
        leg['iv'] = 0.0
    else:
        # ---- Expiration dropdown (from yfinance chain) ----
        if exp_options:
            labels = []
            for e, d in exp_options:
                disp_date = datetime.strptime(e, "%Y-%m-%d").strftime("%b %d, %Y")
                labels.append(f"{disp_date} ({d} days)")
            current_idx = 0
            for idx, (e, _) in enumerate(exp_options):
                if e == leg.get('expiration'):
                    current_idx = idx
                    break
            sel = cols[3].selectbox(
                "", labels, index=current_idx,
                key=f"exp_{i}", label_visibility="collapsed",
            )
            chosen_idx = labels.index(sel)
            leg['expiration'] = exp_options[chosen_idx][0]
            leg['dte'] = exp_options[chosen_idx][1]
        else:
            new_dte = cols[3].number_input(
                "", min_value=0, value=int(leg['dte']), step=1,
                key=f"dte_{i}", label_visibility="collapsed",
            )
            leg['dte'] = new_dte
            leg['expiration'] = (datetime.now().date() + timedelta(days=new_dte)).strftime("%Y-%m-%d")

        # ---- Type FIRST (affects which strikes the chain offers) ----
        leg['type'] = cols[5].selectbox(
            "", ['call', 'put'],
            index=(0 if leg['type'] == 'call' else 1),
            key=f"typ_{i}", label_visibility="collapsed",
        )

        # ---- Strike: SELECTBOX of REAL chain strikes for (exp, type) ----
        chain_strikes = list_strikes(ticker, leg['expiration'], leg['type']) if leg['expiration'] else []
        strk_key = f"strk_{i}"
        if chain_strikes:
            # Snap current strike to nearest available if it isn't in the chain
            current_strike = float(leg.get('strike', 0))
            if not any(abs(s - current_strike) < 0.01 for s in chain_strikes):
                snapped = min(chain_strikes, key=lambda s: abs(s - current_strike))
                leg['strike'] = snapped
                # Also clear widget state so the selectbox picks up the snapped value,
                # and force re-sync since strike changed.
                if strk_key in st.session_state:
                    del st.session_state[strk_key]
                leg.pop('_synced_key', None)
            # Find index for selectbox default
            try:
                strk_idx = next(j for j, s in enumerate(chain_strikes)
                                if abs(s - float(leg['strike'])) < 0.01)
            except StopIteration:
                strk_idx = 0
            leg['strike'] = cols[4].selectbox(
                "", chain_strikes, index=strk_idx,
                key=strk_key, label_visibility="collapsed",
                format_func=lambda s: f"${s:.2f}",
            )
        else:
            # Chain unavailable — fall back to free-form number input
            leg['strike'] = cols[4].number_input(
                "", value=float(leg['strike']), step=0.5,
                key=strk_key, label_visibility="collapsed", format="%.2f",
            )

        # ---- Auto-sync entry/IV from chain when (exp, strike, type) changed ----
        # Pass spot/r/iv_from_mid so the helper can compute IV via BS inverse
        _sync_leg_to_chain(leg, ticker, use_mid, leg_idx=i,
                           spot=spot, r=r, iv_from_mid=iv_from_mid)

        # ---- Entry price input (chain value populated by auto-sync) ----
        px_key = f"px_{i}"
        new_entry = cols[6].number_input(
            "", value=float(leg['entry_price']), step=0.01, format="%.2f",
            key=px_key, label_visibility="collapsed",
        )
        # Detect manual override (user typed a value different from chain mid)
        chain_ref_price = (leg.get('_chain_mid') if use_mid else leg.get('_chain_last')) or 0
        if (leg.get('_chain_status') == 'ok' and chain_ref_price > 0
                and abs(new_entry - chain_ref_price) > 0.005):
            leg['_manual_edit'] = True
        else:
            leg['_manual_edit'] = False
        leg['entry_price'] = new_entry

        # Source indicator under the entry field
        status = leg.get('_chain_status', 'na')
        if status == 'ok':
            bid, ask = leg.get('_chain_bid', 0), leg.get('_chain_ask', 0)
            mid, last = leg.get('_chain_mid', 0), leg.get('_chain_last', 0)
            vol, oi = leg.get('_chain_volume', 0), leg.get('_chain_oi', 0)
            if leg.get('_manual_edit'):
                badge = (f"<span style='color:#b48cff;font-size:0.7rem'>"
                         f"✏️ manual (chain mid ${mid:.2f}, last ${last:.2f})</span>")
            else:
                badge = (f"<span style='color:#00c290;font-size:0.7rem' "
                         f"title='vol {vol} · OI {oi}'>📡 chain: "
                         f"bid ${bid:.2f} / ask ${ask:.2f} / mid ${mid:.2f} / last ${last:.2f}"
                         f"</span>")
        elif status == 'no_chain':
            badge = "<span style='color:#f0b90b;font-size:0.7rem'>⚠️ no chain available</span>"
        elif status == 'no_strike':
            cs = leg.get('_closest_strike')
            badge = (f"<span style='color:#f0b90b;font-size:0.7rem'>"
                     f"⚠️ not in chain (nearest ${cs:.2f})</span>" if cs
                     else "<span style='color:#f0b90b;font-size:0.7rem'>⚠️ not in chain</span>")
        elif status == 'no_price':
            badge = "<span style='color:#f0b90b;font-size:0.7rem'>⚠️ no quote (bid/ask/last all 0)</span>"
        else:
            badge = ""
        if badge:
            cols[6].markdown(badge, unsafe_allow_html=True)

        # ---- Live delta (always BS-calculated since yfinance has no greeks) ----
        d_val, _, _, _ = bs_greeks(spot, leg['strike'], leg['dte'],
                                    leg['iv'] + iv_shift, r, leg['type'])
        sign_for_delta = 1 if leg['action'] == 'buy' else -1
        cols[7].markdown(f"**{d_val * sign_for_delta:+.3f}**")

        # ---- IV input — DISPLAYED AS PERCENTAGE (e.g. 97.02%) ----
        # Internally stored as decimal (0.9702), but the user sees %
        iv_key = f"iv_{i}"
        iv_pct_display = float(leg['iv']) * 100  # convert to percent for widget
        new_iv_pct = cols[8].number_input(
            "", value=iv_pct_display, step=1.0, format="%.2f",
            key=iv_key, label_visibility="collapsed",
            help="Implied volatility (annualized, %)",
        )
        new_iv = new_iv_pct / 100  # back to decimal for internal storage
        # IV manual edit detection (compare against the IV we adopted from chain)
        chain_iv = leg.get('_chain_iv', 0) or 0
        if (leg.get('_chain_status') == 'ok' and chain_iv > 0
                and abs(new_iv - chain_iv) > 0.0005):
            leg['_manual_edit'] = True
        leg['iv'] = new_iv

        # IV source readout (small text under the IV field)
        if status == 'ok':
            iv_yf = leg.get('_chain_iv_yf', 0) or 0
            iv_comp = leg.get('_chain_iv_computed')
            if iv_from_mid and iv_comp is not None:
                iv_badge = (f"<span style='color:#00c290;font-size:0.7rem' "
                            f"title='Inverted Black-Scholes against mid ${mid:.2f}. "
                            f"yfinance reported {iv_yf*100:.2f}%.'>"
                            f"📡 from mid (yf: {iv_yf*100:.1f}%)</span>")
            elif iv_yf > 0:
                iv_badge = (f"<span style='color:#00c290;font-size:0.7rem'>"
                            f"📡 yfinance IV</span>")
            else:
                iv_badge = ""
            if iv_badge:
                cols[8].markdown(iv_badge, unsafe_allow_html=True)

    if cols[9].button("✕", key=f"rm_{i}"):
        to_remove = i

if to_remove is not None and len(st.session_state.legs) > 1:
    st.session_state.legs.pop(to_remove)
    st.rerun()

bc1, bc2, _ = st.columns([1.2, 1.2, 4])
if bc1.button("➕ Add option leg"):
    front_exp = exp_options[0][0] if exp_options else ''
    front_dte = exp_options[0][1] if exp_options else 30
    # Snap initial strike to nearest ATM in the actual chain
    init_strikes = list_strikes(ticker, front_exp, 'call') if front_exp else []
    init_strike = (min(init_strikes, key=lambda s: abs(s - spot))
                   if init_strikes else round(spot, 2))
    st.session_state.legs.append({
        'action': 'buy', 'qty': 1, 'type': 'call', 'strike': init_strike,
        'expiration': front_exp, 'dte': front_dte,
        'entry_price': 1.00, 'iv': realized_vol,
    })
    st.rerun()
if bc2.button("📦 Add shares"):
    st.session_state.legs.append({
        'action': 'buy', 'qty': 100, 'type': 'stock', 'strike': 0,
        'expiration': '', 'dte': 0, 'entry_price': spot, 'iv': 0,
    })
    st.rerun()

# --------------- ANALYZE-AT-DATE SLIDER -----------------------------
option_legs = [l for l in st.session_state.legs if l['type'] != 'stock']
if not option_legs:
    st.warning("Add at least one option leg to see the analysis.")
    st.stop()

min_dte = min(l['dte'] for l in option_legs)
section("Time machine", "Analyze at date")
ac1, ac2 = st.columns([3, 1])
analyze_days = ac1.slider(
    f"Days forward from today (0 = today, {min_dte} = earliest leg's expiration)",
    0, max(min_dte, 1), min(16, min_dte), step=1,
    help="Move forward in time to see how the P&L surface evolves. The filled "
         "curve is your P&L at this date; the dashed curve is always at expiration.",
)
analyze_date_str = (datetime.now().date() + timedelta(days=analyze_days)).strftime("%b %d, %Y")
ac2.markdown(
    f"<div style='margin-top:1.2rem; background:#171c26; border:1px solid #262d3d; "
    f"border-radius:6px; padding:0.55rem 0.9rem;'>"
    f"<div class='small-cap'>Analyze date</div>"
    f"<div class='mono' style='font-size:1.05rem;'>{analyze_date_str}</div>"
    f"</div>", unsafe_allow_html=True)

# --------------- COMPUTE METRICS ------------------------------------
metrics = compute_metrics(
    st.session_state.legs, spot, sigma_default,
    analyze_days_elapsed=analyze_days, iv_shift=iv_shift, r=r,
)

# View range
view_map = {"±20%": 0.20, "±40%": 0.40, "±60%": 0.60, "±100%": 1.00, "All": 2.5}
vr = view_map[view_choice]
view_low = max(0.01, spot * (1 - vr))
view_high = spot * (1 + vr)

# --------------- CHART ----------------------------------------------
section("Payoff", "Profit &amp; Loss")
fig = build_pnl_chart(
    metrics, spot, sigma_default, analyze_days,
    ticker, view_low, view_high,
    show_prob=show_prob, r=r,
)
st.plotly_chart(fig, use_container_width=True)

# --------------- TRADE INFORMATION ----------------------------------
section("Metrics", "Trade Information")

# Position summary string
def _format_leg_summary(leg, ticker):
    action_word = "BUY" if leg['action'] == 'buy' else "SELL"
    sign = "+" if leg['action'] == 'buy' else "-"
    color = C_GREEN if leg['action'] == 'buy' else C_RED
    if leg['type'] == 'stock':
        body = f"{sign}{leg['qty']} {ticker} shares"
    else:
        exp_label = ""
        if leg.get('expiration'):
            try:
                exp_label = datetime.strptime(leg['expiration'], "%Y-%m-%d").strftime("%b %d, %Y") + " "
            except Exception:
                pass
        body = (f"{sign}{leg['qty']} {ticker} {exp_label}"
                f"{leg['strike']:.2f} {leg['type']}")
    return color, action_word, body

summary_parts = []
for leg in st.session_state.legs:
    color, action_word, body = _format_leg_summary(leg, ticker)
    summary_parts.append(
        f"<div class='mono' style='color:{color};'>"
        f"{action_word} {body} @${leg['entry_price']:.2f}</div>"
    )
st.markdown(
    "<div class='strategy-card'>"
    "<div class='small-cap' style='margin-bottom:0.35rem;'>Custom strategy</div>"
    + "".join(summary_parts) +
    "</div>", unsafe_allow_html=True)

# Stock + Trade Details rows
st.markdown("##### Stock")
s1, s2 = st.columns(2)
s1.metric(f"{ticker} Current Price", f"${spot:.2f}",
          help="Latest close from yfinance (editable in the sidebar).")
# Use analyze-date breakevens for the headline (matches the solid curve on chart).
# Fall back to at-expiration breakevens if the analyze-date curve has none.
display_bes = metrics.get('breakevens_analyze') or metrics['breakevens']
display_pnl = metrics['pnl_at_analyze'] if metrics.get('breakevens_analyze') else metrics['pnl_at_exp']
_be_help = ("Price where the position's P&L crosses zero on the analyze date "
            "(the solid curve). Crossing it puts the position in profit. "
            "See Glossary below for the formula context.")
if not display_bes:
    s2.metric(f"{ticker} Breakeven Price", "—", "Not in viewable range",
              help=_be_help)
elif len(display_bes) == 1:
    be = display_bes[0]
    direction = "Above" if display_pnl[-1] > 0 else "Below"
    s2.metric(f"{ticker} Breakeven Price",
              f"{direction} ${be:.2f}", f"{(be/spot-1)*100:+.2f}%",
              help=_be_help)
else:
    be_lo, be_hi = display_bes[0], display_bes[-1]
    inside = display_pnl[0] < 0 and display_pnl[-1] < 0
    s2.metric(f"{ticker} Breakeven Range",
              f"${be_lo:.2f} – ${be_hi:.2f}",
              f"profit {'inside' if inside else 'outside'} range",
              help=_be_help)

st.markdown("##### Trade Details")
td1, td2, td3, td4 = st.columns(4)
if metrics['cost'] >= 0:
    td1.metric("Cost of Trade (debit)", f"${metrics['cost']:,.2f}",
               help="Net premium paid to open: Σ (buy premiums − sell premiums) "
                    "× 100 × qty. Positive = you pay (debit).")
else:
    td1.metric("Credit Received", f"${abs(metrics['cost']):,.2f}",
               help="Net premium collected at open: Σ (sell premiums − buy "
                    "premiums) × 100 × qty. You keep this if all short legs "
                    "expire worthless.")
mp_str = f"${metrics['max_profit']:,.0f}" if metrics['max_profit'] < 1e6 else "Unlimited"
td2.metric("Maximum Profit", mp_str, f"at ${metrics['max_profit_price']:.2f}",
           help="Highest P&L on the at-expiration curve across the modeled "
                "price range. 'Unlimited' = keeps growing past the range edge.")
ml_str = f"-${abs(metrics['max_loss']):,.0f}" if metrics['max_loss'] > -1e6 else "Unlimited"
td3.metric("Maximum Loss", ml_str, f"at ${metrics['max_loss_price']:.2f}",
           delta_color="inverse",
           help="Lowest P&L on the at-expiration curve across the modeled "
                "price range. For multi-expiration positions this includes the "
                "cost of buying back legs that still have time value.")
td4.metric("CVaR (5%)", f"${metrics['cvar']:,.0f}",
           help="Conditional Value at Risk: the AVERAGE P&L across the worst "
                "5% of price outcomes at the earliest expiration. A tail-risk "
                "measure — harsher than max loss probability alone.")

st.markdown("##### Probability Analysis (at earliest expiration)")
p1, p2, p3 = st.columns(3)
p1.metric("Probability of Profit", f"{metrics['pop']*100:.1f}%",
          help="P(P&L > 0 at the earliest expiration): the lognormal "
               "probability mass over all price regions where the expiration "
               "curve is positive. Model-based, not a guarantee.")
p2.metric("Probability of Max Profit", f"{metrics['prob_max_profit']*100:.1f}%",
          help="Probability that price at the earliest expiration lands in the "
               "region where P&L is within 0.5% of the maximum profit. For "
               "unbounded strategies (long calls/puts) this is the probability "
               "of reaching the edge of the modeled price range.")
p3.metric("Probability of Max Loss", f"{metrics['prob_max_loss']*100:.1f}%",
          help="Probability that price at the earliest expiration lands in the "
               "region where P&L is within 0.5% of the maximum loss.")
p4, p5, p6 = st.columns(3)
T_imp = metrics['min_dte'] / 365.0
im_1sigma = implied_move(spot, sigma_default, T_imp)
p4.metric("1σ implied move", f"±${im_1sigma:.2f}",
          f"${spot-im_1sigma:.2f} – ${spot+im_1sigma:.2f}",
          help="One-standard-deviation price move by the earliest expiration: "
               "S × σ × √T. The market 'expects' price to stay inside this "
               "band ~68% of the time.")
p5.metric("Expected Value", f"${metrics['expected_value']:,.0f}",
          help="Probability-weighted average P&L: ∫ P&L(S) × f(S) dS over the "
               "lognormal density f. Positive EV = profitable on average "
               "under the model's assumptions.")
p6.metric("Expected Return", f"{metrics['expected_return_pct']:.1f}%",
          help="Expected Value ÷ Capital at Risk — EV per dollar you're "
               "putting on the line.")

st.markdown("##### Risk / Reward")
rr1, rr2, rr3 = st.columns(3)
rr_str = "∞" if metrics['reward_risk'] == float('inf') else f"{metrics['reward_risk']:.2f}"
rr1.metric("Reward / Risk", rr_str,
           help="Maximum profit ÷ |maximum loss|. Above 1 means you stand to "
                "make more than you risk — but check the probabilities too.")
rr2.metric("Capital at Risk", f"${metrics['capital_at_risk']:,.0f}",
           help="The most you can lose: |max loss|. For credit trades this is "
                "the margin you're effectively risking.")
rr3.metric("Cost basis %",
           f"{metrics['cost']/max(abs(metrics['max_loss']),1)*100:+.1f}%",
           help="Net debit (or credit) ÷ |max loss|. Negative = you were paid "
                "to take the position.")

st.markdown("##### Position Greeks (at spot, now)")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta (Δ)", f"{metrics['delta']:+.2f}",
          help="Share-equivalent exposure: position P&L change per $1 move in "
               "the underlying. −32.7 behaves like being short ~33 shares.")
g2.metric("Gamma (Γ)", f"{metrics['gamma']:+.4f}",
          help="Rate of change of delta per $1 underlying move. High |gamma| "
               "= your direction exposure shifts fast as price moves.")
g3.metric("Theta (Θ)", f"${metrics['theta']:+.2f}/day",
          help="Time decay: P&L change per calendar day, all else equal. "
               "Positive = you collect time value (typical for net sellers).")
g4.metric("Vega (ν)", f"${metrics['vega']:+.2f}/1% IV",
          help="Volatility exposure: P&L change if IV moves 1 percentage "
               "point. Negative = an IV spike hurts the position.")

# ====================================================================
# OPTION PRICE HISTORY  (per-contract chart like OptionCharts)
# ====================================================================

section("Time series", "Option Price History")
st.caption(
    "How each contract's price has moved as the underlying moved. Green/red "
    "candles are **real trades** from yfinance's contract history; the dotted "
    "grey line is the **theoretical Black-Scholes price** (at the leg's current "
    "IV) so you can see fair value even on days the contract didn't trade. "
    "The blue line is the underlying (right axis)."
)

opt_legs_idx = [i for i, l in enumerate(st.session_state.legs) if l['type'] != 'stock']
if not opt_legs_idx:
    st.info("Add an option leg to see its price history.")
else:
    hist_period = st.radio(
        "Window", ["1M", "3M", "6M", "1Y"], index=3, horizontal=True,
        key="opt_hist_period",
    )
    period_days_map = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}
    n_days = period_days_map[hist_period]

    tab_labels = []
    for i in opt_legs_idx:
        l = st.session_state.legs[i]
        exp_disp = (datetime.strptime(l['expiration'], "%Y-%m-%d").strftime("%b %d")
                    if l.get('expiration') else "?")
        tab_labels.append(f"{l['action'].upper()} {exp_disp} ${l['strike']:.2f} {l['type'][0].upper()}")

    hist_tabs = st.tabs(tab_labels)
    for tab, i in zip(hist_tabs, opt_legs_idx):
        leg = st.session_state.legs[i]
        with tab:
            if not leg.get('expiration'):
                st.warning("No expiration set for this leg.")
                continue
            occ = occ_symbol(ticker, leg['expiration'], leg['strike'], leg['type'])
            opt_hist = get_option_history(ticker, leg['expiration'],
                                          leg['strike'], leg['type'])
            fig_h, n_real = build_option_history_chart(
                hist, opt_hist, leg, ticker, r=r, period_days=n_days,
            )
            if fig_h is None:
                st.warning("No underlying history available to build the chart.")
                continue
            st.plotly_chart(fig_h, use_container_width=True)
            if n_real == 0:
                st.info(
                    f"No trade history found for contract `{occ}` — it may be "
                    "newly listed or illiquid. Showing theoretical Black-Scholes "
                    "price only (dotted line). Note: the theoretical line uses "
                    "TODAY's IV across the whole window, so it won't capture "
                    "past IV changes."
                )
            else:
                st.caption(
                    f"Contract `{occ}` — {n_real} trading days with real data. "
                    "Gaps between candles = days with no trades. Where candles "
                    "deviate from the dotted line, the market priced the option "
                    "rich/cheap vs. today's IV (or IV itself was different then)."
                )

# ====================================================================
# WHAT-IF SIMULATOR  (price × DTE table, leg breakdown, assignment)
# ====================================================================

section("Scenarios", "What-If Simulator — Assignment &amp; Roll")
st.caption(
    "Pick any price and any day between now and expiration to see exactly what "
    "happens if you (a) close the position at theoretical prices, (b) hold to "
    "expiration, or (c) get assigned. Also includes a roll planner."
)

with st.container(border=True):
    wc1, wc2, wc3 = st.columns([2, 2, 1.6])
    target_price = wc1.slider(
        "Target underlying price ($)",
        float(view_low), float(view_high),
        float(spot), step=0.25,
        key="wi_price",
    )
    target_days = wc2.slider(
        "Days forward from today",
        0, max(min_dte, 1),
        min(analyze_days, min_dte), step=1,
        key="wi_days",
    )
    target_date_str = (datetime.now().date() + timedelta(days=target_days)).strftime("%b %d, %Y")
    wc3.markdown(
        f"<div style='padding-top:1.5rem;'><b>Target date:</b><br>{target_date_str}<br>"
        f"<span class='small-cap'>Spot move: {(target_price/spot-1)*100:+.1f}%</span></div>",
        unsafe_allow_html=True,
    )

    scen = evaluate_scenario(st.session_state.legs, target_price, target_days,
                              iv_shift=iv_shift, r=r)

    # Probability of reaching this price by target date
    T_target = target_days / 365.0
    if T_target > 0 and sigma_default > 0:
        # Use a band of ±5% around the target for "near this price"
        band = max(target_price * 0.025, 0.50)
        p_near = prob_in_range(target_price - band, target_price + band,
                                spot, sigma_default, T_target, r) * 100
        p_above = (1 - float(lognormal_cdf(target_price, spot, sigma_default,
                                            T_target, r))) * 100
        p_below = 100 - p_above
    else:
        p_near = p_above = p_below = 0.0

    # Headline scenario outcomes
    so1, so2, so3, so4 = st.columns(4)
    so1.metric(
        "Close-now P&L",
        f"${scen['close_pnl']:+,.0f}",
        help="P&L if you closed every leg at its theoretical Black-Scholes "
             "value at the target price and date.",
    )
    so2.metric(
        "Hold-to-expiration P&L",
        f"${scen['exp_pnl']:+,.0f}",
        help="P&L if every leg is held to its own expiration and exits at "
             "intrinsic value, assuming spot is at target price at the "
             "earliest expiration.",
    )
    so3.metric(
        f"P(near ${target_price:.2f} ±{band:.2f})",
        f"{p_near:.1f}%",
        f"P>: {p_above:.1f}%  P<: {p_below:.1f}%",
        help=f"Risk-neutral lognormal probability that spot is within "
             f"±${band:.2f} of target on {target_date_str}.",
    )
    so4.metric(
        "Days remaining",
        f"{min_dte - target_days}d",
        f"to earliest expiration",
    )

    # Per-leg breakdown table
    st.markdown("**Per-leg breakdown at target scenario**")
    df_rows = []
    for row in scen['rows']:
        df_rows.append({
            "Leg": row['leg'],
            "Position": row['spec'],
            "Entry $": f"${row['entry']:.2f}",
            "Value at scenario": f"${row['value_now']:.2f}",
            "Close-now P&L": f"${row['close_pnl']:+,.0f}",
            "Hold-to-exp P&L": f"${row['exp_pnl']:+,.0f}",
            "ITM at exp?": "🟢 ITM" if row['itm_at_exp'] else "⚪ OTM",
        })
    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    # ---- Assignment scenario ----
    short_itm_legs = [
        (i, leg) for i, leg in enumerate(st.session_state.legs)
        if leg['type'] != 'stock'
        and leg['action'] == 'sell'
        and _is_itm_at_exp(leg, target_price)
    ]
    if short_itm_legs:
        st.markdown("**🔔 Assignment scenario** — your short legs that finish ITM:")
        assign_rows = []
        for i, leg in short_itm_legs:
            if leg['type'] == 'call':
                shares_delivered = leg['qty'] * 100  # you deliver
                intrinsic = (target_price - leg['strike']) * leg['qty'] * 100
                action_text = f"Deliver {shares_delivered} shares @ ${leg['strike']:.2f}"
                action_text2 = (f"If uncovered, buy {shares_delivered} @ "
                                f"${target_price:.2f} → cost ${intrinsic:,.0f}")
                cash_pnl = leg['entry_price'] * leg['qty'] * 100 - intrinsic
            else:  # short put
                shares_received = leg['qty'] * 100
                intrinsic = (leg['strike'] - target_price) * leg['qty'] * 100
                action_text = f"Buy {shares_received} shares @ ${leg['strike']:.2f}"
                action_text2 = (f"Effective cost basis ${leg['strike'] - leg['entry_price']:.2f}/sh "
                                f"(market is ${target_price:.2f})")
                cash_pnl = leg['entry_price'] * leg['qty'] * 100 - intrinsic
            assign_rows.append({
                "Leg": i + 1,
                "Position": _leg_label(leg),
                "What happens": action_text,
                "Mechanics": action_text2,
                "Cash P&L on leg": f"${cash_pnl:+,.0f}",
            })
        st.dataframe(pd.DataFrame(assign_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Cash P&L = premium received − intrinsic value at expiration. "
            "For a short call this is the same as buying back the option at "
            "intrinsic; the only difference is whether you end up with stock "
            "or cash. For a short put assigned, you end up long shares at the "
            "strike (effective basis = strike − premium)."
        )
    else:
        st.info("No short legs are ITM at the target price — no assignment risk in this scenario.")

# ====================================================================
# ROLL PLANNER
# ====================================================================

with st.expander("🔁 Roll Planner — close a leg and open a replacement", expanded=False):
    st.caption(
        "Pick an existing option leg, then describe the replacement contract. "
        "We'll compute the roll credit/debit using current Black-Scholes values "
        "for the close (or live-chain quotes if you fetch them)."
    )

    rollable_idxs = [i for i, l in enumerate(st.session_state.legs) if l['type'] != 'stock']
    if not rollable_idxs:
        st.info("No option legs to roll.")
    else:
        roll_labels = [f"Leg {i+1}: {_leg_label(st.session_state.legs[i])}" for i in rollable_idxs]
        rc1, rc2 = st.columns([2, 1])
        chosen_lbl = rc1.selectbox("Which leg to roll?", roll_labels, key="roll_leg")
        chosen_idx = rollable_idxs[roll_labels.index(chosen_lbl)]
        old_leg = st.session_state.legs[chosen_idx]
        roll_on_day = rc2.number_input(
            "Roll on day (from today)",
            min_value=0, max_value=max(old_leg['dte'], 1),
            value=min(target_days, old_leg['dte']),
            help="The day you execute the roll. Default = What-If target day.",
        )

        # Close price for old leg
        rem_old = max(0, old_leg['dte'] - roll_on_day)
        close_price = float(bs_price(
            target_price, old_leg['strike'], rem_old,
            max(old_leg['iv'] + iv_shift, 0.01), r, old_leg['type']
        ))
        # If you're long, closing = sell at close_price (you receive close_price)
        # If you're short, closing = buy at close_price (you pay close_price)
        if old_leg['action'] == 'buy':
            close_cashflow = close_price * old_leg['qty'] * 100  # received
            close_label = f"Sell to close: +${close_cashflow:,.2f} received"
        else:
            close_cashflow = -close_price * old_leg['qty'] * 100  # paid
            close_label = f"Buy to close: -${abs(close_cashflow):,.2f} paid"
        # Close P&L on the old leg = close_cashflow + initial_premium_cashflow
        if old_leg['action'] == 'buy':
            init_cf = -old_leg['entry_price'] * old_leg['qty'] * 100
        else:
            init_cf = old_leg['entry_price'] * old_leg['qty'] * 100
        close_pnl = init_cf + close_cashflow

        # New leg picker
        st.markdown("**Replacement contract**")
        nc1, nc2, nc3, nc4 = st.columns([1.8, 1.0, 1.0, 1.0])
        if exp_options:
            new_labels = [f"{datetime.strptime(e, '%Y-%m-%d').strftime('%b %d, %Y')} ({d}d)"
                          for e, d in exp_options]
            new_exp_lbl = nc1.selectbox("New expiration", new_labels, key="roll_exp")
            new_exp_idx = new_labels.index(new_exp_lbl)
            new_exp_str = exp_options[new_exp_idx][0]
            new_exp_dte_now = exp_options[new_exp_idx][1]
        else:
            new_exp_dte_now = nc1.number_input("New expiration (days from now)",
                                                value=old_leg['dte'] + 30, min_value=1)
            new_exp_str = (datetime.now().date() + timedelta(days=new_exp_dte_now)).strftime("%Y-%m-%d")

        # ---- Type FIRST so strike list reflects the right side ----
        new_type = nc4.selectbox(
            "Type", ['call', 'put'],
            index=(0 if old_leg['type'] == 'call' else 1),
            key="roll_type",
        )
        # ---- New strike: chain selectbox (fall back to number_input if no chain) ----
        roll_chain_strikes = list_strikes(ticker, new_exp_str, new_type) if new_exp_str else []
        if roll_chain_strikes:
            default_strike = min(roll_chain_strikes,
                                 key=lambda s: abs(s - float(old_leg['strike'])))
            try:
                default_idx = next(j for j, s in enumerate(roll_chain_strikes)
                                   if abs(s - default_strike) < 0.01)
            except StopIteration:
                default_idx = 0
            new_strike = nc2.selectbox(
                "New strike", roll_chain_strikes, index=default_idx,
                key="roll_strike", format_func=lambda s: f"${s:.2f}",
            )
        else:
            new_strike = nc2.number_input(
                "New strike", value=float(old_leg['strike']), step=0.5, format="%.2f",
                key="roll_strike",
            )
        new_action = nc3.selectbox(
            "Action", ['buy', 'sell'],
            index=(0 if old_leg['action'] == 'buy' else 1),
            key="roll_action",
        )

        # New contract price at roll date
        new_rem_dte_at_roll = max(1, new_exp_dte_now - roll_on_day)
        # Prefer LIVE chain quote for the new contract's IV; fall back to old leg's IV
        roll_quote = lookup_option_quote(ticker, new_exp_str, new_strike, new_type) if new_exp_str else None
        if roll_quote and roll_quote.get('found') and roll_quote.get('iv', 0) > 0:
            new_iv = roll_quote['iv']
            new_iv_source = f"chain IV {new_iv*100:.1f}%"
        else:
            new_iv = old_leg['iv']
            new_iv_source = "old leg IV (no chain quote)"
        new_price_at_roll = float(bs_price(
            target_price, new_strike, new_rem_dte_at_roll,
            max(new_iv + iv_shift, 0.01), r, new_type
        ))
        if new_action == 'buy':
            new_cashflow = -new_price_at_roll * old_leg['qty'] * 100  # paid
            new_label = f"Buy to open: -${abs(new_cashflow):,.2f} paid"
        else:
            new_cashflow = new_price_at_roll * old_leg['qty'] * 100  # received
            new_label = f"Sell to open: +${new_cashflow:,.2f} received"

        # Roll net = close_cashflow + new_cashflow
        roll_net = close_cashflow + new_cashflow

        st.markdown("**Roll mechanics**")
        rmc1, rmc2, rmc3 = st.columns(3)
        rmc1.metric("Close existing leg", close_label.split(": ")[1])
        rmc2.metric("Open new leg", new_label.split(": ")[1])
        roll_label = f"+${roll_net:,.2f} CREDIT" if roll_net >= 0 else f"-${abs(roll_net):,.2f} DEBIT"
        rmc3.metric("Roll net cashflow", roll_label,
                    delta=("Credit" if roll_net >= 0 else "Debit"),
                    delta_color=("normal" if roll_net >= 0 else "inverse"))

        st.markdown("**Position after roll**")
        # Build hypothetical new position
        new_legs = list(st.session_state.legs)
        # The old leg is closed (remove it)
        new_legs = [l for j, l in enumerate(new_legs) if j != chosen_idx]
        # Add the new leg
        new_legs.append({
            'action': new_action,
            'qty': old_leg['qty'],
            'type': new_type,
            'strike': new_strike,
            'expiration': new_exp_str,
            'dte': new_exp_dte_now,
            'entry_price': new_price_at_roll,  # roll-in price = entry basis
            'iv': new_iv,
        })
        new_metrics = compute_metrics(new_legs, spot, sigma_default,
                                       analyze_days_elapsed=analyze_days,
                                       iv_shift=iv_shift, r=r)
        if new_metrics:
            rp1, rp2, rp3, rp4 = st.columns(4)
            mp_new = (f"${new_metrics['max_profit']:,.0f}"
                      if new_metrics['max_profit'] < 1e6 else "Unlimited")
            ml_new = (f"-${abs(new_metrics['max_loss']):,.0f}"
                      if new_metrics['max_loss'] > -1e6 else "Unlimited")
            mp_old = (f"${metrics['max_profit']:,.0f}"
                      if metrics['max_profit'] < 1e6 else "Unlimited")
            ml_old = (f"-${abs(metrics['max_loss']):,.0f}"
                      if metrics['max_loss'] > -1e6 else "Unlimited")
            rp1.metric("New Max Profit", mp_new, f"was {mp_old}")
            rp2.metric("New Max Loss", ml_new, f"was {ml_old}")
            new_be_str = (", ".join(f"${b:.2f}" for b in new_metrics['breakevens'])
                          if new_metrics['breakevens'] else "—")
            rp3.metric("New Breakeven(s)", new_be_str)
            rp4.metric("New PoP", f"{new_metrics['pop']*100:.1f}%",
                       f"was {metrics['pop']*100:.1f}%")
            st.caption(
                "Note: The new entry price uses the **theoretical Black-Scholes value at "
                "the roll date and target spot price**. Real-world fills will differ. "
                "Click *Fetch from option chain* after applying a roll to snap to live mid prices."
            )

            apply_roll = st.button("✅ Apply this roll to my position")
            if apply_roll:
                st.session_state.legs = new_legs
                st.success("Roll applied. The new leg is now part of your position above.")
                st.rerun()

# ====================================================================
# PRICE × DATE P&L TABLE  (OptionStrat-style "profit table")
# ====================================================================

with st.expander("📋 Price × Date P&L Table (close-now values across scenarios)",
                  expanded=False):
    st.caption(
        "Each cell is your P&L if you close every leg at theoretical Black-Scholes "
        "prices on that date with the underlying at that price. Read across to see "
        "how time decay helps or hurts; read down to see how price moves matter."
    )
    n_prices = 9
    n_days = min(7, max(min_dte, 1) + 1)
    prices = np.linspace(spot * 0.85, spot * 1.15, n_prices)
    day_step = max(1, min_dte // (n_days - 1)) if n_days > 1 else 1
    days_grid = list(range(0, min_dte + 1, day_step))[:n_days]
    if days_grid[-1] != min_dte:
        days_grid.append(min_dte)
    # Build matrix
    grid = []
    for d in days_grid:
        row = {}
        row['Day'] = (f"Today" if d == 0 else f"+{d}d  ({(datetime.now().date()+timedelta(days=d)).strftime('%b %d')})")
        pnl_row = position_pnl(st.session_state.legs, prices, days_elapsed=d,
                                iv_shift=iv_shift, r=r)
        for p, v in zip(prices, pnl_row):
            row[f"${p:.2f}"] = f"${v:+,.0f}"
        grid.append(row)
    df_grid = pd.DataFrame(grid)

    # Styler: color positive green, negative red
    def color_cell(val):
        if not isinstance(val, str) or not val.startswith('$'):
            return ''
        try:
            v = float(val.replace('$', '').replace(',', '').replace('+', ''))
        except ValueError:
            return ''
        if v > 0:
            return 'background-color: rgba(0,194,144,0.12); color: #00c290;'
        if v < 0:
            return 'background-color: rgba(246,70,93,0.12); color: #f6465d;'
        return ''
    styled = df_grid.style.map(color_cell, subset=df_grid.columns[1:])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ====================================================================
# GLOSSARY & FORMULAS
# ====================================================================

section("Reference", "Glossary &amp; Formulas")
st.caption("Every metric on this page, defined. Hover the ❔ on any metric card for the short version.")

gcol1, gcol2 = st.columns(2)

with gcol1:
    with st.expander("📐 Pricing model — Black-Scholes"):
        st.markdown(
            "All theoretical values use the **Black-Scholes** model. "
            "A call is worth:"
        )
        st.latex(r"C = S\,N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)\,T}{\sigma\sqrt{T}},\qquad d_2 = d_1 - \sigma\sqrt{T}")
        st.markdown(
            "- **S** spot price · **K** strike · **T** years to expiration · "
            "**r** risk-free rate · **σ** implied volatility · **N(·)** "
            "standard normal CDF\n"
            "- Puts follow from put-call parity: "
            "$P = C - S + Ke^{-rT}$\n"
            "- **Implied volatility (IV)** is the σ that makes the BS price "
            "equal the market price — this app inverts the formula against "
            "the bid/ask mid (Brent's method) when 'Compute IV from chain "
            "mid' is on."
        )

    with st.expander("🎲 Probability model — lognormal"):
        st.markdown(
            "Probabilities assume the standard risk-neutral lognormal price "
            "model (the same assumption inside Black-Scholes):"
        )
        st.latex(r"S_T = S_0 \exp\!\Big[\big(r - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big],\quad Z\sim\mathcal{N}(0,1)")
        st.markdown(
            "- **Probability of Profit (PoP)** — total probability mass over "
            "every price region where the at-expiration P&L is positive.\n"
            "- **P(max profit) / P(max loss)** — probability mass over the "
            "regions where P&L is within 0.5% of its extreme (the flat "
            "plateau of a spread, or the tail beyond a long option's range).\n"
            "- **1σ implied move** — $S_0\\,\\sigma\\sqrt{T}$: the one-standard-"
            "deviation band; price stays inside it ≈68% of the time, inside "
            "2σ ≈95%.\n"
            "- ⚠️ These are *model* probabilities: they ignore fat tails, "
            "IV changes, and earnings jumps."
        )

    with st.expander("💰 Trade economics"):
        st.markdown(
            "- **Cost of Trade / Credit Received** — net premium at open: "
            "$\\sum (\\text{buys} - \\text{sells}) \\times 100 \\times qty$. "
            "Debit = you pay; credit = you collect.\n"
            "- **Maximum Profit / Loss** — the extremes of the at-expiration "
            "P&L curve over the modeled price range. For positions whose "
            "legs expire on different dates, the value of later-dated legs "
            "at the earliest expiration is their Black-Scholes value "
            "(remaining time value included).\n"
            "- **Breakeven** — where the P&L curve crosses zero. The "
            "headline number uses the analyze-date (solid) curve; the chart "
            "marks it in amber.\n"
            "- **Unlimited** — the curve keeps rising/falling at the edge of "
            "the modeled range (e.g. long call upside, naked short call "
            "risk)."
        )

with gcol2:
    with st.expander("📊 Risk metrics"):
        st.markdown(
            "- **Expected Value (EV)** — probability-weighted average "
            "P&L:"
        )
        st.latex(r"EV = \int P\&L(S)\, f(S)\, dS")
        st.markdown(
            "  where $f$ is the lognormal density at the earliest "
            "expiration.\n"
            "- **Expected Return** — EV ÷ Capital at Risk.\n"
            "- **Reward / Risk** — max profit ÷ |max loss|.\n"
            "- **Capital at Risk** — |max loss|; what you can actually "
            "lose.\n"
            "- **CVaR (5%)** — *Conditional Value at Risk*: the average "
            "P&L across the worst 5% of modeled outcomes. Unlike max loss "
            "(a single point), CVaR weights the whole bad tail.\n"
            "- **Cost basis %** — net debit/credit ÷ |max loss|; negative "
            "means you were paid to open."
        )

    with st.expander("🧮 The Greeks (position-level)"):
        st.markdown(
            "Sensitivities of the whole position, summed across legs "
            "(× qty × 100, shorts negated):\n\n"
            "- **Delta (Δ)** $= \\partial V/\\partial S$ — P&L per \\$1 "
            "underlying move. Also the share-equivalent: Δ = −33 trades "
            "like short 33 shares. Per-leg Δ (in the editor) is also the "
            "rough probability the option expires ITM.\n"
            "- **Gamma (Γ)** $= \\partial^2 V/\\partial S^2$ — how fast "
            "delta changes; high near the strike close to expiration.\n"
            "- **Theta (Θ)** $= \\partial V/\\partial t$ — P&L per "
            "calendar day from time decay alone. Sellers want it "
            "positive.\n"
            "- **Vega (ν)** $= \\partial V/\\partial \\sigma$ — P&L per "
            "1-percentage-point IV change. Short options = negative vega "
            "= IV spikes hurt.\n\n"
            "Greeks here are **calculated** via Black-Scholes from the "
            "chain's IV (yfinance doesn't supply them)."
        )

    with st.expander("🔄 Rolls & assignment"):
        st.markdown(
            "- **Roll** — close an existing leg and open a replacement at "
            "a different strike and/or expiration in one move. The Roll "
            "Planner prices the close at Black-Scholes theoretical value "
            "and the new leg at its live chain quote; **roll credit** = "
            "premium received − cost to close.\n"
            "- **Assignment** — a short option that finishes ITM is "
            "assigned: short calls deliver stock at the strike (called "
            "away), short puts receive stock at the strike. The What-If "
            "simulator flags short legs that are ITM at the scenario "
            "price.\n"
            "- **DTE** — days to expiration. **OCC symbol** — the "
            "exchange contract code (e.g. `MSTR270115C00200000`) used to "
            "fetch per-contract history."
        )


st.markdown("---")
st.caption(
    "Theoretical values from Black-Scholes; probabilities from a risk-neutral "
    "lognormal model. Real outcomes can differ due to IV regime shifts, early "
    "assignment, fill prices, transaction costs, and dividends (not modeled). "
    "Inspired by [optioncharts.io](https://optioncharts.io) and "
    "[optionstrat.com](https://optionstrat.com)."
)
st.markdown(
    "<div style='text-align:center; color:#8b93a7; font-size:0.8rem; padding:8px 0;'>"
    "Created by <b>Thuan</b> · in collaboration with "
    "<b>Claude</b> (Anthropic) · for educational purposes — not financial advice"
    "</div>",
    unsafe_allow_html=True,
)
