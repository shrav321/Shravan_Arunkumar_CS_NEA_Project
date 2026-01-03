# test_render_findings_test3_erroneous_missing_key.py

from analysis import Render_Findings

ctx = {"expiry": "2030-01-01", "type": "C", "strike": 100.0}
inputs = {"mu": 0.1, "sigma": 0.2, "r": 0.05, "T_years": 1.0, "steps": 252, "N": 1000, "seed": 42}

metrics = {
    "mc_mean": 1.23,
    "mc_median": 1.10,
    "q05": 0.10,
    "q95": 3.40,
    "p_itm": 0.62,
    "premium_ref": 0.50,
    "count": 1000,
}

try:
    Render_Findings(ctx, inputs, metrics)
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
