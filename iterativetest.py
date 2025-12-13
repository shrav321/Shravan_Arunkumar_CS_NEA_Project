# test_ensure_contract_exists_test3_erroneous.py

from db_init import init_db
from market import ensure_contract_exists

print("TEST 3: Erroneous case")

init_db()

try:
    ensure_contract_exists("TSLA_2026-01-16_410.00_X", "TSLA", "2026-01-16", 410, "X")
    print("FAIL: No error raised for invalid option type.")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()


