# test_add_funds_test3_erroneous.py

from db_init import init_db, set_cash, get_cash
from trades import add_funds

print("TEST 3: Erroneous case")

init_db()
set_cash(100.0)

try:
    _ = add_funds(0)
    print("FAIL: No error raised for amount = 0")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

print("Stored balance after failed attempt:", get_cash())
print()
