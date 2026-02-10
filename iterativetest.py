# test_db_basic.py >

from db_init import init_db, get_trades_for_contract
from db_init import insert_contract, insert_trade 

def test_invalid_contract_id_safety():
    print("=== Test 3: Erroneous case (malformed contract id) ===")

    init_db()

    # SQL injection
    bad_id = "AAPL_2025-12-19_100.00_C'; DROP TABLE TRADE; —"

    # Attempt to query trades for this id
    rows = get_trades_for_contract(bad_id)

    print("Result of querying malformed contract id:")
    print(rows)  

    # Ensure TRADE table still exists
    try:
        safe_rows = get_trades_for_contract("AAPL_2025-12-19_100.00_C")
        print("TRADE table still operational.")
    except Exception as e:
        print("Error: TRADE table may have been affected.")
        print(e)

if __name__ == "__main__":
    test_invalid_contract_id_safety()