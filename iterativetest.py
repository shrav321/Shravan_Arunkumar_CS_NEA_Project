# test_unrealised_pl_for_contract_test3_erroneous.py

from portfolio import unrealised_pl_for_contract

print("Test 3: Erroneous case - average_cost None for non-flat position")
print("To test that missing cost basis is rejected for a position with net_quantity > 0.")
print()

position = {
    "net_quantity": 1,
    "average_cost": None
}

try:
    _ = unrealised_pl_for_contract(position, current_price=3.20)
    print("FAIL: No error raised for average_cost None")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()

