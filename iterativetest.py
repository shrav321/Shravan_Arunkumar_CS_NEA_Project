# test_derive_metrics_test3_erroneous_empty.py

from analysis import Derive_Metrics

ctx = {"type": "C", "premium_ref": 1.0}
inputs = {"r": 0.05, "T_years": 1.0}
sim = {"discounted_payoffs": []}

try:
    Derive_Metrics(ctx, inputs, sim)
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
