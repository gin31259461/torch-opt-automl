import pandas as pd
import pytest

from torch_opt_automl.feature import FeatureTypes, identify_feature_types

test_cases = [
    # Test case 1: Dates in 'YYYY-MM-DD' format
    (
        pd.DataFrame({"col1": ["2023-01-01", "2023-01-03", "2023-01-04"]}),
        {"col1": "time_series"},
        {}
    ),
    # Test case 2: Mixed strings and numbers (treated as categorical)
    (
        pd.DataFrame({"col1": ["A", "B", 1, "D", 2]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 3: Numerical data with some missing values
    (
        pd.DataFrame({"col1": [1, 2, None, 4, 5]}),
        {"col1": "numerical"},
        {}
    ),
    # Test case 4: Purely numerical data
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5]}),
        {"col1": "numerical"},
        {}
    ),
    # Test case 5: String data with some missing values
    (
        pd.DataFrame({"col1": ["A", "B", None, "D", "E"]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 6: Purely string data
    (
        pd.DataFrame({"col1": ["A", "B", "C", "D", "E"]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 7: Mixed strings and None values
    (
        pd.DataFrame({"col1": ["A", "B", None, "A", "C"], "col2": [1, 2, 3, 4, 5]}),
        {"col1": "categorical", "col2": "numerical"},
        {}
    ),
    # Test case 8: Empty DataFrame
    (
        pd.DataFrame(),
        {},
        {}
    ),
    # Test case 9: All None values
    (
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        {"col1": "none", "col2": "none"},
        {}
    ),
    # Test case 10: Single value columns
    (
        pd.DataFrame({"col1": [1], "col2": ["A"]}),
        {"col1": "numerical", "col2": "categorical"},
        {}
    ),
    # Test case 11: Large DataFrame with mixed types
    (
        pd.DataFrame(
            {
                "col1": [i for i in range(1000)],
                "col2": [str(i) for i in range(1000)],
                "col3": [i * 1.1 for i in range(1000)],
            }
        ),
        {"col1": "numerical", "col2": "categorical", "col3": "numerical"},
        {}
    ),
    # Test case 12: Dates with different formats
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023/01/02", "2023.01.03"],
                "col2": [1, 2, 3],
            }
        ),
        {"col1": "categorical", "col2": "numerical"},
        {}
    ),
    # Test case 13: Mixed numerical and string types in one column
    (
        pd.DataFrame({"col1": [1, "2", 3, "4", 5]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 14: Boolean column
    (
        pd.DataFrame({"col1": [True, False, True, True, False]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 15: Column with leading/trailing spaces
    (
        pd.DataFrame({"col1": [" A ", "B", " C ", "D", "E "]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 16: Column with only one unique value
    (
        pd.DataFrame({"col1": ["A", "A", "A", "A", "A"]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 17: Time series with missing values
    (
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ]
            }
        ),
        {"col1": "time_series"},
        {}
    ),
    # Test case 18: Categorical column with high cardinality
    (
        pd.DataFrame({"col1": [f"category_{i}" for i in range(1000)]}),
        {"col1": "categorical"},
        {}
    ),
    # Test case 19: Numerical column with pre-identified type as categorical
    (
        pd.DataFrame({"col1": [1, 2, 1, 1, 3], "col2": [1.1, 2, 3, 4, 5]}),
        {"col1": "categorical", "col2": "numerical"},
        {"col1": FeatureTypes.categorical},
    ),
    # Test case 20: Numerical column pre-identified as categorical
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["A", "B", "A", "C", "B"],
                "col3": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ],
                ),
            }
        ),
        {"col1": "categorical", "col2": "categorical", "col3": "time_series"},
        {"col1": FeatureTypes.categorical},
    ),
]


# Parametrize the test function to run on each test case
@pytest.mark.parametrize("input_df, expected_types, pre_identified", test_cases)
def test_identify_feature_types(input_df, expected_types, pre_identified):
    result_types = identify_feature_types(input_df, pre_identified)
    assert result_types == expected_types
