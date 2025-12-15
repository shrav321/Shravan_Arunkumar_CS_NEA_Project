# test_get_current_option_price_test3_erroneous.py

from market import get_current_option_price

print("TEST 3: Erroneous case")

row = {"ask": 0, "lastPrice": None}

try:
    _ = get_current_option_price(row)
    print("FAIL: No error raised when price is unusable.")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()



