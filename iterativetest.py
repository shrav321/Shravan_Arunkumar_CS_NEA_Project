# test_attach_greeks_to_portfolio_view_test3_erroneous_missing_sigma.py

from portfolio import attach_greeks_to_portfolio_view

print("Test 3: Erroneous case - missing sigma raises a clear error")
print("Type: Erroneous")

positions = [
    {"contract_id": "D", "ticker": "TSLA", "expiry": "2030-12-31", "strike": 100.0, "type": "C", "net_quantity": 1},
]

spot_by_ticker = {"TSLA": 200.0}
sigma_by_ticker = {}  # sigma deliberately missing for TSLA

try:
    attach_greeks_to_portfolio_view(positions, spot_by_ticker, sigma_by_ticker)
    raise AssertionError("Expected ValueError was not raised")
except ValueError as e:
    assert "Missing sigma for ticker" in str(e)
    print(f"Caught error: {e}")

print("Result: PASS")
