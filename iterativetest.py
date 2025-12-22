# test_execute_exercise_from_portfolio_test3_erroneous.py

from db_init import init_db, set_cash, insert_contract
from trades import execute_buy_from_market, execute_exercise_from_portfolio

print("Test 3: Erroneous case - OTM option")
print("To test that an OTM option is rejected at exercise when a valid open position exists.")
print("Type: Erroneous")
print()

init_db()
set_cash(1000.0)

cid = "AAPL_2025-01-01_150.00_C"
insert_contract(cid, "AAPL", "2025-01-01", 150.0, "C")

option_row = {
    "ticker": "AAPL",
    "expiry": "2025-01-01",
    "strike": 150.0,
    "type": "C",
    "ask": 2.00,
    "lastPrice": 2.00
}

# Create an open position first
execute_buy_from_market(option_row, 1)

contract = {
    "contract_id": cid,
    "expiry": "2025-01-01",
    "strike": 150.0,
    "type": "C"
}

# Spot below strike makes a call out-of-the-money
try:
    _ = execute_exercise_from_portfolio(
        contract,
        underlying_spot=140.0,
        current_date="2025-01-01"
    )
    print("FAIL: No error raised for out-of-the-money option")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print()
