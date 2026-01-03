# test_load_contract_context_test3_erroneous_insufficient_closes.py

from db_init import init_db, insert_price_row
from analysis import Load_Contract_Context

init_db()

ticker = "ONEPT"
insert_price_row(ticker, "2025-01-01", 50.0)

try:
    Load_Contract_Context(
        ticker="ONEPT",
        expiry="2030-01-01",
        strike=50.0,
        option_type="C",
        spot=55.0
    )
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
