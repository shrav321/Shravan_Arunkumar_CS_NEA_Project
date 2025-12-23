# test_compute_mispricing_for_contract_test3_erroneous.py

from market import compute_mispricing_for_contract

print("Test 3: Erroneous case - missing option_row rejected")
print("Type: Erroneous\n")

try:
    _ = compute_mispricing_for_contract(None, spot=100.0, sigma=0.2)
    print("FAIL")
except ValueError as e:
    print("PASS:", e)

print()
