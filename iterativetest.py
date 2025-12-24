# test_compute_bs_greeks_for_contract_test3_erroneous.py

from portfolio import compute_bs_greeks_for_contract

print("Test 3: Erroneous case - expired contract rejected")
print("Type: Erroneous\n")

contract = {"strike": 100.0, "expiry": "2000-01-01", "type": "C"}

try:
    _ = compute_bs_greeks_for_contract(contract, spot=105.0, sigma=0.2)
    print("FAIL")
except ValueError as e:
    print("PASS:", e)
print()
