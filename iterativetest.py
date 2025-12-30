# test_build_model_inputs_test3_erroneous.py

from analysis import Build_Model_Inputs

ctx = {
    "valid": True,
    "ticker": "TEST",
    "expiry": "2030-01-01",
    "closes": [100],
}

Build_Model_Inputs(ctx)

