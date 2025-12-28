# test_build_portfolio_view_with_risk_metrics_test3_erroneous.py

from db_init import init_db, set_cash, insert_contract, insert_trade
import portfolio

init_db()
set_cash(0.0)

cid = "MSFT_2030-01-01_100.00_C"
insert_contract(cid, "MSFT", "2030-01-01", 100.0, "C")
insert_trade(cid, 1, 2.0, "2025-01-01T00:00:00+00:00", "BUY")

all_trades = [
    (1, cid, 1, 2.0, "2025-01-01T00:00:00+00:00", "BUY")
]

try:
    portfolio.build_portfolio_view_with_risk_metrics(
        all_trades,
        current_prices={},  # missing cid
        spot_by_ticker={"MSFT": 120.0},
        sigma_by_ticker={"MSFT": 0.20}
    )
    raise AssertionError("Expected ValueError due to missing current price was not raised.")
except ValueError as e:
    assert "Missing current price for contract_id" in str(e)

print("PASS: Missing current price triggers a clear ValueError.")
