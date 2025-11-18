from db_init import init_db, get_cash, set_cash, adjust_cash

def test_cash_initial_and_adjust():
    init_db()
    print("Initial cash:", get_cash())
    set_cash(1000.0)
    adjust_cash(-250.0)
    print("Cash after debit:", get_cash())


if __name__ == "__main__":
    test_cash_initial_and_adjust()

