# test_execute_buy_from_market_test3_erroneous.py

from db_init import init_db, set_cash, get_cash, get_connection
from trades import execute_buy_from_market

print("TEST 3: Erroneous case")

init_db()
set_cash(1000.0)

option_row = {
    "ticker": "AAPL",
    "expiry": "2026-01-16",
    "strike": 150.0,
    "type": "C",
    "ask": 2.50
}

cash_before = get_cash()

try:
    _ = execute_buy_from_market(option_row, 0)
    print("FAIL: No error raised for quantity = 0")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

cash_after = get_cash()

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM TRADE;")
total_trades = cur.fetchone()[0]
conn.close()

print("Cash before:", cash_before)
print("Cash after:", cash_after)
print("Total trades in table:", total_trades)
print()
