# test_attach_contract_details_test3_erroneous.py

from db_init import init_db
from portfolio import attach_contract_details

print("TEST 3: Erroneous case")

init_db()

holdings = {"MISSING_CONTRACT_ID": 1}

try:
    _ = attach_contract_details(holdings)
    print("FAIL: No error raised for missing contract_id")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()

