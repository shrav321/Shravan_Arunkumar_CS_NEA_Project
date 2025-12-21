# cycle2_integration_test_buy_sell_holdings.py

from db_init import init_db, set_cash, get_cash, get_trades_for_contract
from trades import execute_buy_from_market, execute_sell_from_portfolio, CONTRACT_MULTIPLIER
from portfolio import derive_holdings_from_trade


def main():
    print("CYCLE 2 INTEGRATION TEST - BUY, BUY, SELL, HOLDINGS")
    print()

    # 1) Initialise database and reset cash
    init_db()
    set_cash(100000.0)
    print("Step 1 - DB initialised, cash set to:", get_cash())
    print()

    # 2) Execute first buy
    option_row_buy_1 = {
        "ticker": "AAPL",
        "expiry": "2026-01-16",
        "strike": 150.0,
        "type": "C",
        "ask": 2.50,
        "lastPrice": 2.40
    }

    cash_before_buy1 = get_cash()
    buy1 = execute_buy_from_market(option_row_buy_1, 2)
    cash_after_buy1 = get_cash()

    expected_cost_buy1 = 2.50 * 2 * CONTRACT_MULTIPLIER

    print("Step 2 - BUY 1 executed")
    print("Contract:", buy1["contract_id"])
    print("Cash before BUY 1:", cash_before_buy1)
    print("Cash after BUY 1:", cash_after_buy1)
    print("Expected cash after BUY 1:", cash_before_buy1 - expected_cost_buy1)
    print()

    # 3) Execute second buy (same contract, different premium)
    option_row_buy_2 = {
        "ticker": "AAPL",
        "expiry": "2026-01-16",
        "strike": 150.0,
        "type": "C",
        "ask": 3.00,
        "lastPrice": 2.95
    }

    cash_before_buy2 = get_cash()
    buy2 = execute_buy_from_market(option_row_buy_2, 1)
    cash_after_buy2 = get_cash()

    expected_cost_buy2 = 3.00 * 1 * CONTRACT_MULTIPLIER

    print("Step 3 - BUY 2 executed")
    print("Contract:", buy2["contract_id"])
    print("Cash before BUY 2:", cash_before_buy2)
    print("Cash after BUY 2:", cash_after_buy2)
    print("Expected cash after BUY 2:", cash_before_buy2 - expected_cost_buy2)
    print()

    cid = buy1["contract_id"]

    # 4) Execute sell (sell 1 contract)
    option_row_sell = {
        "ask": 3.20,
        "lastPrice": 3.10
    }

    cash_before_sell = get_cash()
    sell1 = execute_sell_from_portfolio(cid, 1, option_row_sell)
    cash_after_sell = get_cash()

    expected_proceeds_sell = 3.20 * 1 * CONTRACT_MULTIPLIER

    print("Step 4 - SELL executed")
    print("Contract:", sell1["contract_id"])
    print("Cash before SELL:", cash_before_sell)
    print("Cash after SELL:", cash_after_sell)
    print("Expected cash after SELL:", cash_before_sell + expected_proceeds_sell)
    print()

    # 5) Reconstruct holdings from trades
    trades_for_contract = get_trades_for_contract(cid)
    holdings = derive_holdings_from_trade(trades_for_contract)
    net_qty = holdings.get(cid, 0)

    print("Step 5 - Holdings reconstructed from trades")
    print("Trades for contract:", len(trades_for_contract))
    print("Holdings dict:", holdings)
    print("Net quantity for contract:", net_qty)
    print("Expected net quantity:", 2)  # bought 2, bought 1, sold 1 = 2
    print()

    print("CYCLE 2 INTEGRATION TEST COMPLETE")


if __name__ == "__main__":
    main()
