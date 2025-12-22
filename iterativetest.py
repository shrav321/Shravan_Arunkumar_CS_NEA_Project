# test_build_portfolio_view_with_pl_test3_erroneous.py

from db_init import init_db, insert_contract
from portfolio import build_portfolio_view_with_pl

print("Test 3: Erroneous case - missing price for an open position")
print("To test that missing current price entries are rejected.")
print("Type: Erroneous")
print()

init_db()

cid = "AAPL_2026-01-16_150.00_C"
insert_contract(cid, "AAPL", "2026-01-16", 150.0, "C")

all_trades = [
    (1, cid, 1, 2.50, "t1", "BUY"),
]

try:
    _ = build_portfolio_view_with_pl(all_trades, current_prices={}, contract_multiplier=100)
    print("FAIL: No error raised for missing current price")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()
