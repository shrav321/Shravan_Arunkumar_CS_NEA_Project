# test_compute_historical_volatility_test3_erroneous.py

from market import compute_historical_volatility

print("Test 3: Erroneous case - invalid ticker")
print("Type: Erroneous\n")

try:
    compute_historical_volatility("")
    print("FAIL")
except ValueError as e:
    print("PASS:", e)

print()
