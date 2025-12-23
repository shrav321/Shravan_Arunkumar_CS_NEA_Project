# test_bs_price_yf_test3_erroneous.py

from market import bs_price_yf

print("Test 3: Erroneous case - expired option rejected")
print("Type: Erroneous\n")

option_row = {"strike": 100.0, "expiry": "2000-01-01", "type": "C"}

try:
    _ = bs_price_yf(option_row, spot=105.0, sigma=0.2, r=0.01)
    print("FAIL")
except ValueError as e:
    print("PASS:", e)

print()
