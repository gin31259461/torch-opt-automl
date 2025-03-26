from pathlib import Path

import pandas as pd

from torch_opt_automl.feature import FeatureTypes, data_transformation

df = pd.DataFrame(
    {
        "time_series_feature": pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        ),
        "numerical_feature": [1, 2, 3, 4, 5],
        "categorical_feature": ["A", "B", "A", "C", "B"],
    }
)

feature_types = {
    "time_series_feature": FeatureTypes.time_series,
    "numerical_feature": FeatureTypes.numerical,
    "categorical_feature": FeatureTypes.categorical,
}

transformed_df = data_transformation(
    df, feature_types, numerical_transformation="normalization"
)

transformed_df.to_csv(f"{Path().parent.parent.absolute()}/csv/test.csv", index=False)
