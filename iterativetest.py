# test_fetch_options_by_ticker_and_type_test3_erroneous_bad_type.py

from market import fetch_options_by_ticker_and_type


class _Ticker:
    def __init__(self):
        self.options = ["2030-01-01"]

    def option_chain(self, expiry):
        return None


def _provider(_symbol):
    return _Ticker()


print("Test 3: Erroneous case - invalid option type rejected")
print("Type: Erroneous")

try:
    fetch_options_by_ticker_and_type("AAPL", "X", ticker_provider=_provider)
    raise AssertionError("Expected ValueError was not raised")
except ValueError as e:
    msg = str(e).lower()
    assert "option_type" in msg or "must be" in msg

print("Result: PASS")
