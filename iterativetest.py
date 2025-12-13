# Test 3: Erroneous case – unsupported option type raising a ValueError

from market import build_contract_id

print("TEST 3: Erroneous case")
try:
    cid = build_contract_id("TSLA", "2026-01-16", 410, "X")
    print("No error raised. Output:", cid)   # Incorrect behaviour
except ValueError as e:
    print("Error caught as expected:", e)
print()
