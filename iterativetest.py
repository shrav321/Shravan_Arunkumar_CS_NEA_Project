# test_get_position_metrics_test3_erroneous.py

from portfolio import get_position_metrics

print("TEST 3: Erroneous case")

cid = "MSFT_2026-01-16_300.00_C"

trades = [
    (1, cid, 1, 2.00, "t1", "BUY"),
    (2, cid, 2, 2.50, "t2", "SELL"),
]

try:
    _ = get_position_metrics(trades)
    print("FAIL: No error raised for negative net quantity")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()


