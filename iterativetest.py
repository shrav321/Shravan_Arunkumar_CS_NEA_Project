# test_visualise_results_test3_erroneous_missing_field.py

from analysis import Visualise_Results

sim = {
    "paths_subset": [[1.0, 2.0, 3.0]],
}

try:
    Visualise_Results(sim)
    raise AssertionError("Expected ValueError was not raised")
except ValueError:
    pass
