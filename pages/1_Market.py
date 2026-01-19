# pages/1_Market.py

import streamlit as st

from db_init import init_db, get_cash, adjust_cash, get_db_path
from market import build_contract_id, fetch_options_by_ticker_and_type
from trades import execute_buy_from_market, execute_sell_from_portfolio


DEV_MODE = False


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


def _read_cash_safely() -> float:
    try:
        return float(get_cash())
    except Exception:
        init_db()
        return float(get_cash())


init_db()
st.set_page_config(page_title="Market", layout="wide")
st.title("Market")

st.markdown(
    """
This page supports live option discovery by ticker and type, followed by buy and sell actions
recorded in the persistent trade ledger.
"""
)

# -------------------------
# Sidebar: System controls
# -------------------------
with st.sidebar:
    st.header("System")
    st.caption(get_db_path())

    if st.button("Re-initialise Database"):
        init_db()
        st.success("Database initialised")
        _rerun()

    st.subheader("Cash")
    cash_now = _read_cash_safely()
    st.metric("Current Cash", f"{cash_now:.2f}")

    if "deposit_amount" not in st.session_state:
        st.session_state.deposit_amount = 0.0

    st.number_input(
        "Deposit Amount",
        min_value=0.0,
        step=100.0,
        key="deposit_amount"
    )

    if st.button("Deposit"):
        try:
            amt = float(st.session_state.deposit_amount)
            if amt <= 0:
                raise ValueError("Deposit amount must be > 0")
            adjust_cash(amt)
            st.success("Cash updated")
            _rerun()
        except Exception as e:
            st.error(str(e))

    if DEV_MODE:
        st.caption("Development controls")
        if "set_cash_value" not in st.session_state:
            st.session_state.set_cash_value = cash_now

        st.number_input(
            "Set Cash Balance",
            min_value=0.0,
            step=100.0,
            key="set_cash_value"
        )
        if st.button("Set Cash"):
            try:
                from db_init import set_cash
                set_cash(float(st.session_state.set_cash_value))
                st.success("Cash set")
                _rerun()
            except Exception as e:
                st.error(str(e))


# -------------------------
# Market discovery controls
# -------------------------
st.header("Live option search")

if "last_query_rows" not in st.session_state:
    st.session_state.last_query_rows = []

if "last_query_meta" not in st.session_state:
    st.session_state.last_query_meta = {}

col1, col2, col3, col4 = st.columns([1.2, 1.0, 1.2, 1.0])

with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

with col2:
    opt_label = st.radio("Type", options=["Call", "Put"], horizontal=True)
    opt_type = "C" if opt_label == "Call" else "P"

with col3:
    expiry_filter = st.text_input("Expiry filter (YYYY-MM-DD or blank)", value="").strip()

with col4:
    fetch_btn = st.button("Fetch chain")

filter_row1, filter_row2, filter_row3 = st.columns(3)

with filter_row1:
    strike_min = st.number_input("Min strike (optional)", min_value=0.0, value=0.0, step=1.0)

with filter_row2:
    strike_max = st.number_input("Max strike (optional)", min_value=0.0, value=0.0, step=1.0)

with filter_row3:
    limit_rows = st.number_input("Max rows", min_value=25, max_value=5000, value=250, step=25)

if fetch_btn:
    try:
        if ticker == "":
            raise ValueError("Ticker must be non-empty")

        rows = fetch_options_by_ticker_and_type(ticker, opt_type)


        filtered = []
        for r in rows:
            try:
                k = float(r.get("strike", 0.0))
            except Exception:
                continue

            if strike_min > 0 and k < float(strike_min):
                continue
            if strike_max > 0 and k > float(strike_max):
                continue

            filtered.append(r)

        if len(filtered) > int(limit_rows):
            filtered = filtered[:int(limit_rows)]

        st.session_state.last_query_rows = filtered
        st.session_state.last_query_meta = {
            "ticker": ticker,
            "type": opt_type,
            "expiry_filter": expiry_filter,
            "strike_min": strike_min,
            "strike_max": strike_max,
            "limit_rows": limit_rows,
        }

        if len(filtered) == 0:
            st.warning("No contracts matched the current filters.")
        else:
            st.success(f"Loaded {len(filtered)} contracts.")
    except Exception as e:
        st.error(str(e))


rows = st.session_state.last_query_rows

if len(rows) == 0:
    st.info("No chain loaded. Use the controls above to fetch live options.")
    st.stop()

# -------------------------
# Display and selection
# -------------------------
st.subheader("Contracts")

display_rows = []
for i, r in enumerate(rows):
    display_rows.append(
        {
            "Index": i,
            "Ticker": r.get("ticker", ""),
            "Expiry": r.get("expiry", ""),
            "Strike": r.get("strike", ""),
            "Type": r.get("type", ""),
            "Ask": r.get("ask", None),
            "Last Price": r.get("lastPrice", None),
            "Bid": r.get("bid", None),
            "Open Interest": r.get("openInterest", None),
            "Implied Vol": r.get("impliedVolatility", None),
        }
    )

st.dataframe(display_rows, use_container_width=True)

selected_index = st.number_input(
    "Select contract index",
    min_value=0,
    max_value=len(rows) - 1,
    step=1,
    value=0,
)

selected = rows[int(selected_index)]

st.subheader("Selected contract")
contract_id = build_contract_id(
    selected["ticker"],
    selected["expiry"],
    float(selected["strike"]),
    selected["type"],
)
st.code(contract_id)

qty = st.number_input("Quantity (contracts)", min_value=1, step=1, value=1)

colA, colB = st.columns(2)

with colA:
    if st.button("Buy selected"):
        try:
            res = execute_buy_from_market(selected, int(qty))
            st.success("Buy executed")
            st.json(res)
            _rerun()
        except Exception as e:
            st.error(str(e))

with colB:
    if st.button("Sell selected"):
        try:
            res = execute_sell_from_portfolio(contract_id, int(qty), selected)
            st.success("Sell executed")
            st.json(res)
            _rerun()
        except Exception as e:
            st.error(str(e))


# -------------------------
# Optional fallback: manual quote board
# -------------------------
with st.expander("Manual quote board (fallback)", expanded=False):
    st.caption(
        "This is useful when live chain data is missing usable prices or for deterministic demonstrations."
    )

    if "board" not in st.session_state:
        st.session_state.board = []

    board = st.session_state.board

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        mt = st.text_input("Board ticker", value="AAPL", key="board_ticker").strip().upper()

    with b2:
        ml = st.radio("Board type", options=["Call", "Put"], horizontal=True, key="board_type")
        mtp = "C" if ml == "Call" else "P"

    with b3:
        me = st.text_input("Board expiry", value="2030-01-01", key="board_expiry").strip()

    with b4:
        ms = st.number_input("Board strike", min_value=0.01, step=1.0, value=100.0, key="board_strike")

    with b5:
        ma = st.number_input("Board ask", min_value=0.0, step=0.1, value=2.0, key="board_ask")

    mlp = st.number_input("Board last price", min_value=0.0, step=0.1, value=1.9, key="board_last")

    cA, cB = st.columns(2)

    with cA:
        if st.button("Add to board"):
            try:
                if mt == "":
                    raise ValueError("Ticker must be non-empty")
                if me == "":
                    raise ValueError("Expiry must be non-empty")

                board.append(
                    {
                        "ticker": mt,
                        "expiry": me,
                        "strike": float(ms),
                        "type": mtp,
                        "ask": float(ma),
                        "lastPrice": float(mlp),
                    }
                )
                st.success("Added")
            except Exception as e:
                st.error(str(e))

    with cB:
        if st.button("Clear board"):
            st.session_state.board = []
            st.success("Cleared")
            _rerun()

    if len(board) > 0:
        st.dataframe(board, use_container_width=True)
