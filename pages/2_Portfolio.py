# pages/2_Portfolio.py

import streamlit as st
from datetime import date
from typing import Dict, Any, List

from market import get_live_option_premium_for_contract
from db_init import init_db, get_cash, get_all_trades, get_trades_for_contract
from portfolio import (
    build_portfolio_view,
    unrealised_pl_for_contract,
    build_portfolio_view_with_risk_metrics,
)
from trades import execute_exercise_from_portfolio, execute_expire_worthless_from_portfolio


# Utility function to force refresh the Streamlit UI state
def _rerun() -> None:
    fn = getattr(st, "rerun", None)
    if callable(fn):
        fn()
        return
    fn = getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()
        return
    raise RuntimeError("No rerun function available in this Streamlit version")


# Utility to fetch cash balance with a database initialization fallback
def _read_cash_safely() -> float:
    try:
        return float(get_cash())
    except Exception:
        init_db()
        return float(get_cash())


# Page configuration and database connection setup
st.set_page_config(page_title="Portfolio", layout="wide")
init_db()

st.title("Portfolio")
st.caption(
    "Positions are reconstructed from the trade ledger. Lifecycle resolution is recorded as explicit trade events."
)

# Sidebar: Displays current account liquidity and manual reload trigger
with st.sidebar:
    st.header("Account")
    st.metric("Cash Balance", f"{_read_cash_safely():.2f}")
    if st.button("Reload"):
        _rerun()

# Load all trade records from the persistent SQL database
try:
    all_trades = get_all_trades()
except Exception as e:
    st.error(str(e))
    st.stop()

# Aggregate trade history into a summary view of active positions
positions = build_portfolio_view(all_trades)

st.subheader("Open positions")

if len(positions) == 0:
    st.info("No open positions.")
    st.stop()

# Display current holdings in a formatted data table
st.dataframe(
    [
        {
            "contract_id": p["contract_id"],
            "ticker": p["ticker"],
            "expiry": p["expiry"],
            "strike": p["strike"],
            "type": p["type"],
            "net_quantity": p["net_quantity"],
            "average_cost": (None if p["average_cost"] is None else round(float(p["average_cost"]), 6)),
        }
        for p in positions
    ],
    use_container_width=True,
)

# Interface for selecting a specific position for further analysis or action
contract_ids = [p["contract_id"] for p in positions]
selected_cid = st.selectbox("Select contract", contract_ids)

selected_position = None
for p in positions:
    if p["contract_id"] == selected_cid:
        selected_position = p
        break

if selected_position is None:
    st.error("Selected contract could not be resolved.")
    st.stop()

st.divider()

col_left, col_right = st.columns(2)

# Calculation of Unrealised Profit and Loss based on live market premiums
with col_left:
    st.subheader("Unrealised P/L")

    if st.button("Fetch live premium and compute P/L"):
        try:
            live_premium = get_live_option_premium_for_contract(
                ticker=selected_position["ticker"],
                expiry=selected_position["expiry"],
                strike=float(selected_position["strike"]),
                option_type=selected_position["type"],
            )
            pl = unrealised_pl_for_contract(selected_position, live_premium)
            st.success(f"Live premium: {live_premium:.4f}")
            st.success(f"Unrealised P/L (selected): {pl:.2f}")
        except Exception as e:
            st.error(str(e))

# Logic for contract lifecycle management: manual exercise or worthless expiry
with col_right:
    st.subheader("Lifecycle resolution")

    current_date = st.date_input("Current date (YYYY-MM-DD)", value=date.today())
    spot_lifecycle = st.number_input(
        "Underlying spot (for lifecycle actions)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.2f",
        key="spot_lifecycle",
    )

    contract = {
        "contract_id": selected_position["contract_id"],
        "expiry": selected_position["expiry"],
        "strike": selected_position["strike"],
        "type": selected_position["type"],
    }

    btn_ex, btn_exp = st.columns(2)

    with btn_ex:
        if st.button("Exercise (ITM only)"):
            try:
                res = execute_exercise_from_portfolio(
                    contract=contract,
                    underlying_spot=float(spot_lifecycle),
                    current_date=str(current_date),
                )
                st.success("Exercise recorded.")
                st.json(res)
                _rerun()
            except Exception as e:
                st.error(str(e))

    with btn_exp:
        if st.button("Expire worthless (OTM only)"):
            try:
                res = execute_expire_worthless_from_portfolio(
                    contract=contract,
                    underlying_spot=float(spot_lifecycle),
                    current_date=str(current_date),
                )
                st.success("Worthless expiry recorded.")
                st.json(res)
                _rerun()
            except Exception as e:
                st.error(str(e))

st.divider()

# Advanced risk assessment section including Black-Scholes Greeks
st.subheader("Risk Metrics")
st.caption(
    "Market premiums are supplied explicitly per contract. Theoretical price, mispricing, and Greeks are derived from the pricing model."
)

# Manage session state for market premiums across all held contracts
if "premium_inputs" not in st.session_state:
    st.session_state.premium_inputs = {}

for cid in contract_ids:
    if cid not in st.session_state.premium_inputs:
        st.session_state.premium_inputs[cid] = 0.0

# Aggregation of tickers for spot/volatility inputs
tickers = sorted({p["ticker"] for p in positions})

if "spot_inputs" not in st.session_state:
    st.session_state.spot_inputs = {}
if "vol_inputs" not in st.session_state:
    st.session_state.vol_inputs = {}

for tkr in tickers:
    if tkr not in st.session_state.spot_inputs:
        st.session_state.spot_inputs[tkr] = 0.0
    if tkr not in st.session_state.vol_inputs:
        st.session_state.vol_inputs[tkr] = 0.0

# Controls for toggling between live market data and manual input overrides
mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 1])
with mode_col1:
    use_live_market_inputs = st.checkbox(
        "Use live spot and volatility",
        value=True,
        help="When enabled, spot and volatility are fetched from the market layer. When disabled, manual inputs are used.",
    )

with mode_col2:
    use_live_premiums = st.checkbox(
        "Auto-fetch live premiums",
        value=False,
        help="When enabled, current market premiums are automatically fetched for all positions. May take a few seconds.",
    )

with mode_col3:
    run_risk = st.button("Compute risk metrics")

# Background fetch of live premiums if automated tracking is enabled
if use_live_premiums:
    with st.spinner("Fetching live premiums for all positions..."):
        try:
            for cid in contract_ids:
                pos = next(p for p in positions if p["contract_id"] == cid)
                live_premium = get_live_option_premium_for_contract(
                    ticker=pos["ticker"],
                    expiry=pos["expiry"],
                    strike=float(pos["strike"]),
                    option_type=pos["type"],
                )
                st.session_state.premium_inputs[cid] = float(live_premium)
            st.success(f"Fetched live premiums for {len(contract_ids)} contracts.")
        except Exception as e:
            st.error(f"Error fetching live premiums: {str(e)}")
            use_live_premiums = False  # Fall back to manual entry mode

# Collapsible manual price input for mispricing analysis
with st.expander("Enter current premiums for open positions", expanded=not use_live_premiums):
    st.write("These premiums are used as the market price for mispricing calculations.")
    for cid in contract_ids:
        st.number_input(
            label=cid,
            min_value=0.0,
            step=0.1,
            format="%.4f",
            key=f"prem_{cid}",
            value=float(st.session_state.premium_inputs[cid]),
        )
        st.session_state.premium_inputs[cid] = float(st.session_state[f"prem_{cid}"])

# Collapsible manual volatility and spot inputs for sensitivity modeling
if not use_live_market_inputs:
    with st.expander("Manual spot and volatility (optional)", expanded=True):
        st.write("Spot is in currency units. Volatility is annualised, such as 0.20 for 20%.")
        for tkr in tickers:
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    label=f"{tkr} spot",
                    min_value=0.0,
                    step=1.0,
                    format="%.2f",
                    key=f"spot_{tkr}",
                    value=float(st.session_state.spot_inputs[tkr]),
                )
                st.session_state.spot_inputs[tkr] = float(st.session_state[f"spot_{tkr}"])
            with c2:
                st.number_input(
                    label=f"{tkr} volatility",
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"vol_{tkr}",
                    value=float(st.session_state.vol_inputs[tkr]),
                )
                st.session_state.vol_inputs[tkr] = float(st.session_state[f"vol_{tkr}"])

risk_positions: List[Dict[str, Any]] = []

# Calculation of Black-Scholes risk metrics for all portfolio holdings
if run_risk:
    try:
        current_prices: Dict[str, float] = {
            cid: float(st.session_state.premium_inputs[cid]) for cid in contract_ids
        }

        spot_by_ticker = None
        sigma_by_ticker = None

        if not use_live_market_inputs:
            spot_by_ticker = {t: float(st.session_state.spot_inputs[t]) for t in tickers}
            sigma_by_ticker = {t: float(st.session_state.vol_inputs[t]) for t in tickers}

        risk_positions = build_portfolio_view_with_risk_metrics(
            all_trades=all_trades,
            current_prices=current_prices,
            spot_by_ticker=spot_by_ticker,
            sigma_by_ticker=sigma_by_ticker,
        )

        st.success("Risk metrics computed.")
    except Exception as e:
        st.error(str(e))

# Display the final risk table including mispricing and Greeks
if risk_positions:
    st.dataframe(
        [
            {
                "contract_id": p["contract_id"],
                "ticker": p["ticker"],
                "expiry": p["expiry"],
                "strike": p["strike"],
                "type": p["type"],
                "net_quantity": p["net_quantity"],
                "market_price": round(float(p["market_price"]), 6),
                "theoretical_price": round(float(p["theoretical_price"]), 6),
                "mispricing_abs": round(float(p["mispricing_abs"]), 6),
                "mispricing_pct": round(float(p["mispricing_pct"]), 6),
                "delta": round(float(p["delta"]), 6),
                "gamma": round(float(p["gamma"]), 6),
                "vega": round(float(p["vega"]), 6),
                "theta": round(float(p["theta"]), 6),
                "spot": round(float(p["spot"]), 6),
                "volatility": round(float(p["volatility"]), 6),
            }
            for p in risk_positions
        ],
        use_container_width=True,
    )

# Reference definitions for Greek terminology
with st.expander("Greeks explained", expanded=False):
    st.markdown(
        """
**Delta (Δ)** Approximate change in option premium for a 1 unit move in the underlying. Calls typically have positive delta; puts typically have negative delta.

**Gamma (Γ)** Rate of change of delta as the underlying moves. Higher gamma means delta changes faster, which increases curvature risk.

**Vega (V)** Approximate change in option premium for a 1.00 change in volatility (in this implementation, vega is per 1.0 volatility unit). Higher vega means the option price is more sensitive to volatility shifts.

**Theta (Θ)** Time decay. This value is reported per day. Negative theta means the option tends to lose value as expiry approaches, holding other inputs fixed.
        """
    )

st.divider()

# Comprehensive trade history logs, filterable by specific contract
st.subheader("Trade history")
scope = st.radio("Scope", ["Selected contract", "All trades"], horizontal=True)

try:
    if scope == "Selected contract":
        rows = get_trades_for_contract(selected_cid)
    else:
        rows = all_trades
except Exception as e:
    st.error(str(e))
    st.stop()

# Final output of the immutable trade ledger
st.dataframe(
    [
        {
            "trade_id": r[0],
            "contract_id": r[1],
            "quantity": r[2],
            "price": r[3],
            "timestamp": r[4],
            "side": r[5],
        }
        for r in rows
    ],
    use_container_width=True,
)