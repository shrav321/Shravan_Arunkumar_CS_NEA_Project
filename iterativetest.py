# test_run_monte_carlo_test3_erroneous.py

from analysis import Run_Monte_Carlo

ctx = {"S": 100.0, "strike": 100.0, "type": "X"}  # invalid
inputs = {
    "mu": 0.05,
    "sigma": 0.2,
    "r": 0.05,
    "T_years": 0.5,
    "steps": 10,
    "N": 10,
    "seed": 1,
}

try:
    Run_Monte_Carlo(ctx, inputs)
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
