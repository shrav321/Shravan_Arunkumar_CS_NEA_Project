# test_build_portfolio_view_test3_erroneous.py

from db_init import init_db
from portfolio import build_portfolio_view

print("Test 3: Erroneous case - trade references missing contract metadata")
print("To test that missing CONTRACT rows are detected and rejected.")
print()

init_db()

missing_cid = "MISSING_2026-01-16_100.00_C"
all_trades = [
    (1, missing_cid, 1, 2.00, "t1", "BUY"),
]

try:
    _ = build_portfolio_view(all_trades)
    print("FAIL: No error raised for missing contract metadata")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()




