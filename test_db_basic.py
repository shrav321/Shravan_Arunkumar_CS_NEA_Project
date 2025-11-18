from db_init import init_db, insert_contract, get_contract, count_contracts

def test_contract_insert_and_unique():
    init_db()
    
    insert_contract("AAPL_2025-12-19_100.00_C", "AAPL", "2025-12-19", 100.0, "C")
    insert_contract("AAPL_2025-12-19_100.00_C", "AAPL", "2025-12-19", 100.0, "C")
    row = get_contract("AAPL_2025-12-19_100.00_C")
    total = count_contracts()
    print("Contract row:", row)
    print("Total contracts:", total)


if __name__ == "__main__":
    test_contract_insert_and_unique()

