import pandas as pd

from torch_opt_automl.feature import FeatureTypes, data_transformation

# Example DataFrame
df = pd.DataFrame(
    {
        "numerical_feature": [1, 2, 3, 4, 5],
        "time_series_feature": pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        ),
        "categorical_feature": ["A", "B", "A", "C", "B"],
    }
)

# Feature types
feature_types = {
    "numerical_feature": FeatureTypes.numerical,
    "time_series_feature": FeatureTypes.time_series,
    "categorical_feature": FeatureTypes.categorical,
}

# Transform the data
transformed_df = data_transformation(
    df, feature_types, numerical_transformation="normalization"
)

print(transformed_df)
