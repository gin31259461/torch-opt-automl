import numpy as np
import pandas as pd
import pytest

from torch_opt_automl.data_utils.data_col_type_parser import (
    ColumnOperation,
    ColumnRecommendation,
    ColumnRecommendations,
    DataColTypeParser,
)
from torch_opt_automl.data_utils.datacleaner import DataCleaner


@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        "numerical_C01": [1, 2, np.nan, 4, 5, np.nan, 7, -2, -1],
        "categorical_C01": ["A", "B", "A", "B", "C", "A", "B", "D", "E"],
        "outlier_C01": [1, 2, 3, 4, 5, 6, 100, 2, 3],
        "negative_C01": [-1, -2, -3, 1, 2, np.nan, 4, -4, -3],
        "datetime_C01": pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
                "2024-01-09",
            ]
        ),
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_parser(sample_data):
    """Fixture to create a DataColTypeParser instance."""
    return DataColTypeParser(sample_data)


@pytest.fixture
def data_cleaner(sample_data, data_parser):
    """Fixture to create a DataCleaner instance."""
    return DataCleaner(sample_data, data_parser)


def test_drop_column(sample_data):
    """Tests the DROP_COLUMN operation."""
    data_copy = sample_data.copy()
    parser = DataColTypeParser(data_copy)
    recommendations = ColumnRecommendations()
    recommendations.add_recommendation(
        "categorical_C01",
        ColumnRecommendation(ColumnOperation.DROP_COLUMN, "test", 1, 1.0),
    )
    cleaner = DataCleaner(data_copy, parser)
    cleaner.recommendations = recommendations
    cleaner.apply_cleaning_recommendations()
    assert "categorical_C01" not in cleaner.df.columns


def test_outlier_removal(sample_data):
    """Tests the outlier removal operation."""
    data_copy = sample_data.copy()
    parser = DataColTypeParser(data_copy)
    recommendations = ColumnRecommendations()
    recommendations.add_recommendation(
        "outlier_C01",
        ColumnRecommendation(ColumnOperation.REMOVE_OUTLIERS, "test", 1, 1.0),
    )
    cleaner = DataCleaner(data_copy, parser)
    cleaner.recommendations = recommendations
    cleaner.apply_cleaning_recommendations()
    assert not cleaner.df["outlier_C01"].equals(
        sample_data["outlier_C01"]
    )  # Verify outlier handling modifies values


def test_log_transform(sample_data):
    """Tests the log transform operation."""
    data_copy = sample_data.copy()
    parser = DataColTypeParser(data_copy)
    recommendations = ColumnRecommendations()

    recommendations.add_recommendation(
        "numerical_C01",
        ColumnRecommendation(ColumnOperation.LOG_TRANSFORM, "test", 1, 1.0),
    )

    recommendations.add_recommendation(
        "negative_C01",
        ColumnRecommendation(ColumnOperation.LOG_TRANSFORM, "test", 1, 1.0),
    )

    cleaner = DataCleaner(data_copy, parser)
    cleaner.recommendations = recommendations
    cleaner.apply_cleaning_recommendations()

    assert not cleaner.df["numerical_C01"].equals(sample_data["numerical_C01"])
    assert not cleaner.df["negative_C01"].equals(sample_data["negative_C01"])
