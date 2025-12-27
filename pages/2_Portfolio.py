# pages/2_Portfolio.py

import streamlit as st
from datetime import date

from db_init import init_db, get_cash, get_all_trades, get_trades_for_contract
from portfolio import build_portfolio_view, unrealised_pl_for_contract
from trades import execute_exercise_from_portfolio, execute_expire_worthless_from_portfolio


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _read_cash_safely() -> float:
    try:
        return float(get_cash())
    except Exception:
        init_db()
        return float(get_cash())


st.set_page_config(page_title="Portfolio", layout="wide")
init_db()

st.title("Portfolio")
st.caption("Positions are reconstructed from the trade ledger and resolved at expiry via explicit lifecycle actions.")

# Sidebar: account + reload
with st.sidebar:
    st.header("Account")
    st.metric("Cash Balance", f"{_read_cash_safely():.2f}")
    if st.button("Reload"):
        _rerun()

# Load trades and reconstruct positions
try:
    all_trades = get_all_trades()
except Exception as e:
    st.error(str(e))
    st.stop()

positions = build_portfolio_view(all_trades)

st.subheader("Open positions")

if len(positions) == 0:
    st.info("No open positions.")
    st.stop()

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

# Contract selector
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

# -----------------------------
# Unrealised P/L (selected only)
# -----------------------------
with col_left:
    st.subheader("Unrealised P/L")

    current_price = st.number_input(
        "Current option premium",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.2f",
    )

    if st.button("Compute P/L for selected"):
        try:
            pl = unrealised_pl_for_contract(selected_position, float(current_price))
            st.success(f"Unrealised P/L (selected): {pl:.2f}")
        except Exception as e:
            st.error(str(e))

# -----------------------------
# Lifecycle resolution
# -----------------------------
with col_right:
    st.subheader("Lifecycle resolution")

    current_date = st.date_input("Current date (YYYY-MM-DD)", value=date.today())
    spot = st.number_input(
        "Underlying spot",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.2f",
    )

    # Build minimal contract dict expected by lifecycle actions
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
                    underlying_spot=float(spot),
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
                    underlying_spot=float(spot),
                    current_date=str(current_date),
                )
                st.success("Worthless expiry recorded.")
                st.json(res)
                _rerun()
            except Exception as e:
                st.error(str(e))

st.divider()

# -----------------------------
# Trade history
# -----------------------------
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
