# test_build_model_inputs_test3_erroneous_invalid_ctx.py

from analysis import Build_Model_Inputs

ctx = {
    "valid": False,
    "ticker": "AAPL",
    "expiry": "2030-01-01",
    "closes": [100.0, 101.0, 102.0],
}

try:
    Build_Model_Inputs(ctx)
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
