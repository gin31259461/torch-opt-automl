import pandas as pd
import pytest

from torch_opt_automl.feature import (
    FeatureTypes,
    clean_identified_df_and_feature_types,
    data_cleaning,
    identify_feature_types,
    transform_identified_df_features,
)

test_identify_feature_types_test_cases = [
    # Test case 1: Dates in 'YYYY-MM-DD' format
    (
        pd.DataFrame({"col1": ["2023-01-01", "2023-01-03", "2023-01-04"]}),
        {"col1": "time_series"},
        {},
    ),
    # Test case 2: Mixed strings and numbers (treated as categorical)
    (pd.DataFrame({"col1": ["A", "B", 1, "D", 2]}), {"col1": "categorical"}, {}),
    # Test case 3: Numerical data with some missing values
    (pd.DataFrame({"col1": [1, 2, None, 4, 5]}), {"col1": "numerical"}, {}),
    # Test case 4: Purely numerical data
    (pd.DataFrame({"col1": [1, 2, 3, 4, 5]}), {"col1": "numerical"}, {}),
    # Test case 5: String data with some missing values
    (pd.DataFrame({"col1": ["A", "B", None, "D", "E"]}), {"col1": "categorical"}, {}),
    # Test case 6: Purely string data
    (pd.DataFrame({"col1": ["A", "B", "C", "D", "E"]}), {"col1": "categorical"}, {}),
    # Test case 7: Mixed strings and None values
    (
        pd.DataFrame({"col1": ["A", "B", None, "A", "C"], "col2": [1, 2, 3, 4, 5]}),
        {"col1": "categorical", "col2": "numerical"},
        {},
    ),
    # Test case 8: Empty DataFrame
    (pd.DataFrame(), {}, {}),
    # Test case 9: All None values
    (
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        {"col1": "none", "col2": "none"},
        {},
    ),
    # Test case 10: Single value columns
    (
        pd.DataFrame({"col1": [1], "col2": ["A"]}),
        {"col1": "numerical", "col2": "categorical"},
        {},
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
        {},
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
        {},
    ),
    # Test case 13: Mixed numerical and string types in one column
    (pd.DataFrame({"col1": [1, "2", 3, "4", 5]}), {"col1": "categorical"}, {}),
    # Test case 14: Boolean column
    (
        pd.DataFrame({"col1": [True, False, True, True, False]}),
        {"col1": "categorical"},
        {},
    ),
    # Test case 15: Column with leading/trailing spaces
    (
        pd.DataFrame({"col1": [" A ", "B", " C ", "D", "E "]}),
        {"col1": "categorical"},
        {},
    ),
    # Test case 16: Column with only one unique value
    (pd.DataFrame({"col1": ["A", "A", "A", "A", "A"]}), {"col1": "categorical"}, {}),
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
        {},
    ),
    # Test case 18: Categorical column with high cardinality
    (
        pd.DataFrame({"col1": [f"category_{i}" for i in range(1000)]}),
        {"col1": "categorical"},
        {},
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


@pytest.mark.parametrize(
    "input_df, expected_types, pre_identified", test_identify_feature_types_test_cases
)
def test_identify_feature_types(input_df, expected_types, pre_identified):
    result_types = identify_feature_types(input_df, pre_identified)
    assert result_types == expected_types


test_transform_identified_df_features_test_cases = [
    # Test Case 1: Time series, numerical, and categorical columns
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
            }
        ),
        pd.DataFrame(
            {
                "col1": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]),
                "col2": [1, 2, 3],
                "col3": pd.Categorical(["A", "B", "C"]),
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.numerical,
            "col3": FeatureTypes.categorical,
        },
    ),
    # Test Case 2: Mixed strings and numbers in one column
    (
        pd.DataFrame({"col1": [1, "2", 3, "4", 5]}),
        pd.DataFrame({"col1": pd.Categorical([1, "2", 3, "4", 5])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 3: Empty DataFrame
    (pd.DataFrame(), pd.DataFrame(), {}),
    # Test Case 4: All None values -> don't do anything
    (
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        {"col1": FeatureTypes.none, "col2": FeatureTypes.none},
    ),
    # Test Case 5: Single value columns
    (
        pd.DataFrame({"col1": [1, 1], "col2": ["A", "A"]}),
        pd.DataFrame({"col1": [1, 1], "col2": pd.Categorical(["A", "A"])}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
    ),
    # Test Case 6: Large DataFrame with mixed types
    (
        pd.DataFrame(
            {
                "col1": [i for i in range(1000)],
                "col2": [str(i) for i in range(1000)],
                "col3": [i * 1.1 for i in range(1000)],
            }
        ),
        pd.DataFrame(
            {
                "col1": [i for i in range(1000)],
                "col2": pd.Categorical([str(i) for i in range(1000)]),
                "col3": [i * 1.1 for i in range(1000)],
            }
        ),
        {
            "col1": FeatureTypes.numerical,
            "col2": FeatureTypes.categorical,
            "col3": FeatureTypes.numerical,
        },
    ),
    # Test Case 7: Dates with different formats that can be converted to datetime
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023/01/02", "2023.01.03"],
                "col2": [1, 2, 3],
            }
        ),
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    ["2023-01-01", "2023/01/02", "2023.01.03"], format="mixed"
                ),
                "col2": [1, 2, 3],
            }
        ),
        {"col1": FeatureTypes.time_series, "col2": FeatureTypes.numerical},
    ),
    # Test Case 8: Boolean column
    (
        pd.DataFrame({"col1": [True, False, True, True, False]}),
        pd.DataFrame({"col1": pd.Categorical([True, False, True, True, False])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 9: Column with leading/trailing spaces
    (
        pd.DataFrame({"col1": [" A ", "B", " C ", "D", "E "]}),
        pd.DataFrame({"col1": pd.Categorical([" A ", "B", " C ", "D", "E "])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 10: Column with only one unique value
    (
        pd.DataFrame({"col1": ["A", "A", "A", "A", "A"]}),
        pd.DataFrame({"col1": pd.Categorical(["A", "A", "A", "A", "A"])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 11: Time series with missing values
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
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        None,
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ),
            }
        ),
        {"col1": FeatureTypes.time_series},
    ),
    # Test Case 12: Categorical column with high cardinality
    (
        pd.DataFrame({"col1": [f"category_{i}" for i in range(1000)]}),
        pd.DataFrame({"col1": pd.Categorical([f"category_{i}" for i in range(1000)])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 13: Numerical column with pre-identified type as categorical
    (
        pd.DataFrame({"col1": [1.1, 2, 3, 4, 5]}),
        pd.DataFrame({"col1": pd.Categorical([1.1, 2, 3, 4, 5])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 14: Numerical column pre-identified as time_series
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5]}),
        pd.DataFrame({"col1": pd.to_datetime([1, 2, 3, 4, 5], errors="coerce")}),
        {"col1": FeatureTypes.time_series},
    ),
    # Test Case 15: DataFrame with only time series columns
    (
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
            }
        ),
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ),
                "col2": pd.to_datetime(
                    [
                        "2023-01-06",
                        "2023-01-07",
                        "2023-01-08",
                        "2023-01-09",
                        "2023-01-10",
                    ]
                ),
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.time_series,
        },
    ),
    # Test Case 16: DataFrame with only categorical columns
    (
        pd.DataFrame(
            {"col1": ["A", "B", "C", "D", "E"], "col2": ["F", "G", "H", "I", "J"]}
        ),
        pd.DataFrame(
            {
                "col1": pd.Categorical(["A", "B", "C", "D", "E"]),
                "col2": pd.Categorical(["F", "G", "H", "I", "J"]),
            }
        ),
        {
            "col1": FeatureTypes.categorical,
            "col2": FeatureTypes.categorical,
        },
    ),
    # Test Case 17: DataFrame with only numerical columns
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]}),
        pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.numerical},
    ),
    # Test Case 18: DataFrame with mixed data types and missing values
    (
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [1, None, 3, 4, 5],
                "col3": ["A", "B", None, "D", "E"],
            }
        ),
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        None,
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ),
                "col2": [1, None, 3, 4, 5],
                "col3": pd.Categorical(["A", "B", None, "D", "E"]),
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.numerical,
            "col3": FeatureTypes.categorical,
        },
    ),
    # Test Case 19: Dates with a format that cannot be converted to datetime
    (
        pd.DataFrame({"col1": ["20230101", "20230103", "20230104"]}),
        pd.DataFrame({"col1": pd.Categorical(["20230101", "20230103", "20230104"])}),
        {"col1": FeatureTypes.categorical},
    ),
    # Test Case 20: Column with pre-identified type as none -> don't do anything
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5]}),
        pd.DataFrame({"col1": [1, 2, 3, 4, 5]}),
        {"col1": FeatureTypes.none},
    ),
]


@pytest.mark.parametrize(
    "input_df, expected_df, feature_types",
    test_transform_identified_df_features_test_cases,
)
def test_transform_identified_df_features(input_df, expected_df, feature_types):
    transformed_df = transform_identified_df_features(input_df, feature_types)
    assert transformed_df.equals(expected_df)


test_clean_identified_df_and_feature_types_test_cases = [
    # Test Case 1: All valid feature types
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.numerical,
            "col3": FeatureTypes.categorical,
        },
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.numerical,
            "col3": FeatureTypes.categorical,
        },
    ),
    # Test Case 2: One 'none' feature type
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.none,
            "col3": FeatureTypes.categorical,
        },
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col3": ["A", "B", "C"],
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col3": FeatureTypes.categorical,
        },
    ),
    # Test Case 3: Multiple 'none' feature types
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
                "col4": [True, False, True],
            }
        ),
        {
            "col1": FeatureTypes.none,
            "col2": FeatureTypes.none,
            "col3": FeatureTypes.categorical,
            "col4": FeatureTypes.categorical,
        },
        pd.DataFrame(
            {
                "col3": ["A", "B", "C"],
                "col4": [True, False, True],
            }
        ),
        {
            "col3": FeatureTypes.categorical,
            "col4": FeatureTypes.categorical,
        },
    ),
    # Test Case 4: All 'none' feature types
    (
        pd.DataFrame(
            {
                "col1": ["2023-01-01", "2023-01-03", "2023-01-04"],
                "col2": [1, 2, 3],
                "col3": ["A", "B", "C"],
            }
        ),
        {
            "col1": FeatureTypes.none,
            "col2": FeatureTypes.none,
            "col3": FeatureTypes.none,
        },
        pd.DataFrame({}),
        {},
    ),
    # Test Case 5: Empty DataFrame
    (pd.DataFrame(), {}, pd.DataFrame({}), {}),
    # Test Case 6: DataFrame with only 'none' columns
    (
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        {"col1": FeatureTypes.none, "col2": FeatureTypes.none},
        pd.DataFrame({}),
        {},
    ),
    # Test Case 7: Single valid feature type
    (
        pd.DataFrame({"col1": [1, 2, 3]}),
        {"col1": FeatureTypes.numerical},
        pd.DataFrame({"col1": [1, 2, 3]}),
        {"col1": FeatureTypes.numerical},
    ),
    # Test Case 8: Single 'none' feature type
    (
        pd.DataFrame({"col1": [1, 2, 3]}),
        {"col1": FeatureTypes.none},
        pd.DataFrame({}),
        {},
    ),
    # Test Case 9: Mixed valid and 'none' feature types
    (
        pd.DataFrame(
            {
                "col1": ["A", "B", "C"],
                "col2": [1, 2, 3],
                "col3": ["2023-01-01", "2023-01-02", "2023-01-03"],
            }
        ),
        {
            "col1": FeatureTypes.categorical,
            "col2": FeatureTypes.none,
            "col3": FeatureTypes.time_series,
        },
        pd.DataFrame(
            {
                "col1": ["A", "B", "C"],
                "col3": ["2023-01-01", "2023-01-02", "2023-01-03"],
            }
        ),
        {"col1": FeatureTypes.categorical, "col3": FeatureTypes.time_series},
    ),
]


@pytest.mark.parametrize(
    "input_df, input_feature_types, expected_df, expected_feature_types",
    test_clean_identified_df_and_feature_types_test_cases,
)
def test_clean_identified_df_and_feature_types(
    input_df, input_feature_types, expected_df, expected_feature_types
):
    (
        cleaned_df,
        cleaned_feature_types,
    ) = clean_identified_df_and_feature_types(input_df, input_feature_types)
    assert cleaned_df.equals(expected_df)
    assert cleaned_feature_types == expected_feature_types


test_data_cleaning_test_cases = [
    # Test Case 1: Numerical column with missing values - mean strategy
    (
        pd.DataFrame({"col1": [1, 2, None, 4, 5], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["A", "B", "C", "D", "E"],
            }
        ),
    ),
    # Test Case 2: Numerical column with missing values - median strategy
    (
        pd.DataFrame({"col1": [1, 2, None, 4, 5], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "median",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["A", "B", "C", "D", "E"],
            }
        ),
    ),
    # Test Case 3: Numerical column with missing values - drop strategy
    (
        pd.DataFrame({"col1": [1, 2, None, 4, 5], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "drop",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 4.0, 5.0],
                "col2": ["A", "B", "D", "E"],
            }
        ),
    ),
    # Test Case 4: Categorical column with missing values - mode strategy
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", None, "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["A", "B", "A", "D", "E"],
            }
        ),
    ),
    # Test Case 5: Categorical column with missing values - drop strategy
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", None, "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "drop",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 4.0, 5.0],
                "col2": ["A", "B", "D", "E"],
            }
        ),
    ),
    # Test Case 6: Time series column with missing values - ffill strategy
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.time_series},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
    ),
    # Test Case 7: Time series column with missing values - bfill strategy
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.time_series},
        "mean",
        "mode",
        "bfill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-04",
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
    ),
    # Test Case 8: Time series column with missing values - drop strategy
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.time_series},
        "mean",
        "mode",
        "drop",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 4.0, 5.0],
                "col2": ["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05"],
            }
        ),
    ),
    # Test Case 9: Numerical column with outliers - iqr method, clip handling
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 100], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        "iqr",
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 7.0],
                "col2": ["A", "B", "C", "D", "E"],
            }
        ),
    ),
    # Test Case 10: Numerical column with outliers - iqr method, drop handling
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 100], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        "iqr",
        "drop",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": ["A", "B", "C", "D"],
            }
        ),
    ),
    # Test Case 11: DataFrame with duplicate rows
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 1],
                "col2": ["A", "B", "C", "D", "E", "A"],
            }
        ),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["A", "B", "C", "D", "E"],
            }
        ),
    ),
    # Test Case 12: Empty DataFrame
    (
        pd.DataFrame(),
        {},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(),
    ),
    # Test Case 13: DataFrame with all columns as 'none' type
    (
        pd.DataFrame({"col1": [None, None], "col2": [None, None]}),
        {"col1": FeatureTypes.none, "col2": FeatureTypes.none},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame({"col1": [None], "col2": [None]}),
    ),
    # Test Case 14: DataFrame with only numerical columns
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.numerical},
        "mean",
        "mode",
        "ffill",
        "iqr",
        "clip",
        1.5,
        pd.DataFrame(
            {"col1": [1.0, 2.0, 3.0, 4.0, 5.0], "col2": [6.0, 7.0, 8.0, 9.0, 10.0]}
        ),
    ),
    # Test Case 15: DataFrame with only categorical columns
    (
        pd.DataFrame(
            {"col1": ["A", "B", "C", "D", "E"], "col2": ["F", "G", "H", "I", "J"]}
        ),
        {"col1": FeatureTypes.categorical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {"col1": ["A", "B", "C", "D", "E"], "col2": ["F", "G", "H", "I", "J"]}
        ),
    ),
    # Test Case 16: DataFrame with only time series columns
    (
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
            }
        ),
        {"col1": FeatureTypes.time_series, "col2": FeatureTypes.time_series},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
            }
        ),
    ),
    # Test Case 17: DataFrame with mixed data types and missing values
    (
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    None,
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [1, None, 3, 4, 5],
                "col3": ["A", "B", None, "D", "E"],
            }
        ),
        {
            "col1": FeatureTypes.time_series,
            "col2": FeatureTypes.numerical,
            "col3": FeatureTypes.categorical,
        },
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "col2": [1.00, 3.25, 3.00, 4.00, 5.00],
                "col3": ["A", "B", "A", "D", "E"],
            }
        ),
    ),
    # Test Case 18: Numerical column with negative outliers - iqr method, clip handling
    (
        pd.DataFrame({"col1": [1, 2, 3, 4, -100], "col2": ["A", "B", "C", "D", "E"]}),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        "iqr",
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, -2.0],
                "col2": ["A", "B", "C", "D", "E"],
            }
        ),
    ),
    # Test Case 19: Time series column with non-date values - should be treated as categorical
    (
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    "Invalid Date",
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
        {"col1": FeatureTypes.numerical, "col2": FeatureTypes.time_series},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [
                    "2023-01-01",
                    "2023-01-02",
                    "Invalid Date",
                    "2023-01-04",
                    "2023-01-05",
                ],
            }
        ),
    ),
    # Test Case 20: Categorical column with mixed data types - should remain categorical
    (
        pd.DataFrame(
            {"col1": [1, "2", 3, "4", True], "col2": ["A", "B", "C", "D", "E"]}
        ),
        {"col1": FeatureTypes.categorical, "col2": FeatureTypes.categorical},
        "mean",
        "mode",
        "ffill",
        None,
        "clip",
        1.5,
        pd.DataFrame(
            {"col1": [1, "2", 3, "4", True], "col2": ["A", "B", "C", "D", "E"]}
        ),
    ),
]


@pytest.mark.parametrize(
    "input_df, feature_types, numerical_missing_strategy, categorical_missing_strategy, time_series_missing_strategy, anomaly_detection_method, anomaly_handling, anomaly_detection_threshold, expected_df",
    test_data_cleaning_test_cases,
)
def test_data_cleaning(
    input_df,
    feature_types,
    numerical_missing_strategy,
    categorical_missing_strategy,
    time_series_missing_strategy,
    anomaly_detection_method,
    anomaly_handling,
    anomaly_detection_threshold,
    expected_df,
):
    cleaned_df = data_cleaning(
        input_df,
        feature_types,
        numerical_missing_strategy,
        categorical_missing_strategy,
        time_series_missing_strategy,
        anomaly_detection_method,
        anomaly_handling,
        anomaly_detection_threshold,
    )

    assert cleaned_df.equals(expected_df)
