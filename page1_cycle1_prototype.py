# page1_cycle1_prototype.py

import streamlit as st

from db_init import init_db, list_tables, get_cash, set_cash, adjust_cash
from market import build_contract_id
from trades import execute_buy_from_market, execute_sell_from_portfolio


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


# Ensure database exists before any reads.
init_db()

st.set_page_config(page_title="Cycle 1 Prototype", layout="wide")

st.title("Cycle 1 Prototype: Executable Market Actions")

st.markdown(
    """
    This prototype demonstrates executable trading actions with persistent state.
    Market discovery is simplified to a small contract board defined inside the interface.
    """
)

# -------------------------
# Sidebar: System controls
# -------------------------
with st.sidebar:
    st.header("System")
    from db_init import get_db_path
    st.caption(get_db_path())


    if st.button("Re-initialise Database"):
        init_db()
        st.success("Database initialised")
        _rerun()

    if st.button("Show Tables"):
        try:
            tables = list_tables()
            st.write(tables)
        except Exception as e:
            st.error(str(e))

    st.subheader("Cash")

    cash_now = _read_cash_safely()
    st.metric("Current Cash", f"{cash_now:.2f}")

    if "deposit_amount" not in st.session_state:
        st.session_state.deposit_amount = 0.0

    if "set_cash_value" not in st.session_state:
        st.session_state.set_cash_value = cash_now

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

    st.number_input(
        "Set Cash Balance",
        min_value=0.0,
        step=100.0,
        key="set_cash_value"
    )

    if st.button("Set Cash"):
        try:
            set_cash(float(st.session_state.set_cash_value))
            st.success("Cash set")
            _rerun()
        except Exception as e:
            st.error(str(e))


# -------------------------
# In-memory contract board
# -------------------------
st.header("Contract Board")

if "board" not in st.session_state:
    st.session_state.board = []

board = st.session_state.board

colA, colB, colC, colD, colE = st.columns(5)

with colA:
    ticker = st.text_input("Ticker", value="AAPL")

with colB:
    opt_label = st.radio("Type", options=["Call", "Put"], horizontal=True)

with colC:
    expiry = st.text_input("Expiry (YYYY-MM-DD)", value="2030-01-01")

with colD:
    strike = st.number_input("Strike", min_value=0.01, step=1.0, value=100.0)

with colE:
    ask = st.number_input("Ask", min_value=0.0, step=0.1, value=2.0)

last_price = st.number_input("Last Price", min_value=0.0, step=0.1, value=1.9)

add_col1, add_col2 = st.columns(2)

with add_col1:
    if st.button("Add Quote To Board"):
        try:
            tkr = str(ticker).strip().upper()
            if tkr == "":
                raise ValueError("Ticker must be non-empty")

            exp = str(expiry).strip()
            if exp == "":
                raise ValueError("Expiry must be non-empty")

            typ = "C" if opt_label == "Call" else "P"

            row = {
                "ticker": tkr,
                "expiry": exp,
                "strike": float(strike),
                "type": typ,
                "ask": float(ask),
                "lastPrice": float(last_price),
            }
            board.append(row)
            st.success("Quote added")
        except Exception as e:
            st.error(str(e))

with add_col2:
    if st.button("Clear Board"):
        st.session_state.board = []
        st.success("Board cleared")
        _rerun()


# -------------------------
# Filtering and selection
# -------------------------
st.subheader("Filter")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    filter_type_label = st.radio(
        "Filter Type",
        options=["Call", "Put"],
        horizontal=True,
        key="filter_type"
    )

with filter_col2:
    expiries = sorted({r["expiry"] for r in board})
    expiry_filter = st.selectbox("Filter Expiry", options=(["All"] + expiries), key="expiry_filter")

filter_type = "C" if filter_type_label == "Call" else "P"

filtered = []
for r in board:
    if str(r.get("type", "")).upper() != filter_type:
        continue
    if expiry_filter != "All" and str(r.get("expiry")) != expiry_filter:
        continue
    filtered.append(r)

if len(filtered) == 0:
    st.info("No quotes available for the current filters.")
else:
    st.subheader("Filtered Quotes")
    st.dataframe(
        [
            {
                "Index": i,
                "Ticker": r["ticker"],
                "Expiry": r["expiry"],
                "Strike": r["strike"],
                "Type": r["type"],
                "Ask": r.get("ask"),
                "Last Price": r.get("lastPrice"),
            }
            for i, r in enumerate(filtered)
        ],
        use_container_width=True,
    )

    selected_index = st.number_input(
        "Select Quote Index",
        min_value=0,
        max_value=len(filtered) - 1,
        step=1,
        value=0,
        key="selected_quote_index"
    )

    selected = filtered[int(selected_index)]

    st.subheader("Selected Contract")
    contract_id = build_contract_id(
        selected["ticker"],
        selected["expiry"],
        float(selected["strike"]),
        selected["type"],
    )
    st.code(contract_id)

    qty = st.number_input("Quantity (contracts)", min_value=1, step=1, value=1, key="trade_qty")

    trade_col1, trade_col2 = st.columns(2)

    with trade_col1:
        if st.button("Buy Selected"):
            try:
                result = execute_buy_from_market(selected, int(qty))
                st.success("Buy executed")
                st.json(result)
                _rerun()
            except Exception as e:
                st.error(str(e))

    with trade_col2:
        if st.button("Sell Selected"):
            try:
                result = execute_sell_from_portfolio(contract_id, int(qty), selected)
                st.success("Sell executed")
                st.json(result)
                _rerun()
            except Exception as e:
                st.error(str(e))
