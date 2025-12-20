# test_derive_holdings_from_trade_test3_erroneous.py

from portfolio import derive_holdings_from_trade

print("TEST 3: Erroneous case")

trades = [
    (1, "AAPL_2026-01-16_150.00_C", 1, 2.00, "t1", "HOLD"),
]

try:
    _ = derive_holdings_from_trade(trades)
    print("FAIL: No error raised for invalid trade side")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()
