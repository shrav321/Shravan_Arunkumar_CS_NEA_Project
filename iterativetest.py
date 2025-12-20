# test_execute_sell_from_portfolio_test3_erroneous.py

from db_init import init_db, set_cash, get_cash, get_connection
from trades import execute_buy_from_market, execute_sell_from_portfolio

print("TEST 3: Erroneous case")

init_db()
set_cash(1000.0)

option_row_buy = {
    "ticker": "AAPL",
    "expiry": "2026-01-16",
    "strike": 150.0,
    "type": "C",
    "ask": 1.00
}
buy_result = execute_buy_from_market(option_row_buy, 1)
cid = buy_result["contract_id"]

cash_before = get_cash()

try:
    _ = execute_sell_from_portfolio(cid, 0, {"ask": 1.10})
    print("FAIL: No error raised for quantity = 0")
except ValueError as e:
    print("PASS: Error caught as expected:", e)

cash_after = get_cash()

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM TRADE WHERE contract_id = ?;", (cid,))
trade_count = cur.fetchone()[0]
conn.close()

print("Cash before:", cash_before)
print("Cash after:", cash_after)
print("Trades recorded for contract:", trade_count)
print()
