import numpy as np


def comparison_percentage(v1, vRef, perc):
    if perc < 0:  # Accept comparation
        return True
    return abs(v1 - vRef) / vRef * 100.0 < perc


def compare_with_expected_perc_tuple(v1, expect_tuple):
    if expect_tuple[1] < 0:  # Accept comparation
        return True
    if np.isclose(expect_tuple[0], 0.0):
        return abs(v1 - expect_tuple[0]) * 100.0 < expect_tuple[1]
    return (
        abs(v1 - expect_tuple[0]) / expect_tuple[0] * 100.0 < expect_tuple[1]
    )


def assert_solution_result(solution, expected):
    # assert np.isclose(DIC, EXPECTED_DIC, 1e-2) #TODO
    assert comparison_percentage(
        solution.pH, expected["pH"][0], expected["pH"][1]
    )
    assert comparison_percentage(
        solution.I, expected["I"][0], expected["I"][1]
    )
    if "sc" in expected:
        assert comparison_percentage(
            solution.sc * 1e6, expected["sc"][0], expected["sc"][1]
        )
