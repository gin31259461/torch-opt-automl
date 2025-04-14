import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import RobustScaler

from torch_opt_automl.data_utils.data_col_type_parser import (
    ColumnOperation,
    ColumnRecommendation,
    DataColTypeParser,
)


class DataCleaner:
    def __init__(self, df: pd.DataFrame, parser: DataColTypeParser):
        self.df = df.copy()
        self.parser = parser
        self.column_types = self.parser.identify_column_types()
        self.recommendations = self.parser.recommend_column_operations()

    def apply_cleaning_recommendations(self):
        """Applies only the cleaning-related recommendations."""
        for col, rec_list in self.recommendations.get_all_recommendations().items():
            for rec in rec_list:
                if rec.operation in [  # Filter for cleaning operations
                    ColumnOperation.DROP_COLUMN,
                    ColumnOperation.IMPUTE_MEAN,
                    ColumnOperation.IMPUTE_MEDIAN,
                    ColumnOperation.IMPUTE_MODE,
                    ColumnOperation.IMPUTE_CONSTANT,
                    ColumnOperation.IMPUTE_KNN,
                    ColumnOperation.IMPUTE_REGRESSION,
                    ColumnOperation.IMPUTE_FORWARD_FILL,
                    ColumnOperation.IMPUTE_BACKWARD_FILL,
                    ColumnOperation.REMOVE_OUTLIERS,
                    ColumnOperation.WINSORIZE_OUTLIERS,
                    ColumnOperation.CAP_OUTLIERS,
                    ColumnOperation.LOG_TRANSFORM,
                ]:  # other data cleaning operations
                    self.apply_cleaning_recommendation(col, rec)

    def apply_cleaning_recommendation(
        self, col: str, recommendation: ColumnRecommendation
    ):
        """Applies a single cleaning recommendation using match/case."""
        operation = recommendation.operation
        params = recommendation.params if recommendation.params else {}

        try:
            match operation:
                case ColumnOperation.DROP_COLUMN:
                    self.df.drop(columns=[col], inplace=True)
                    return

                case ColumnOperation.IMPUTE_MEAN:
                    imputer = SimpleImputer(strategy="mean")
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_MEDIAN:
                    imputer = SimpleImputer(strategy="median")
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_MODE:
                    imputer = SimpleImputer(strategy="most_frequent")
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_CONSTANT:
                    fill_value = params.get("value", 0)  # Default to 0 if not provided
                    imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_KNN:
                    imputer = KNNImputer(**params)
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_REGRESSION:
                    imputer = IterativeImputer(**params)
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                case ColumnOperation.IMPUTE_FORWARD_FILL:
                    self.df[col] = self.df[col].ffill()
                case ColumnOperation.IMPUTE_BACKWARD_FILL:
                    self.df[col] = self.df[col].bfill()

                case ColumnOperation.REMOVE_OUTLIERS:
                    scaler = RobustScaler(quantile_range=(25.0, 75.0), **params)
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                case ColumnOperation.WINSORIZE_OUTLIERS:
                    factor = params.get("factor", 1.5)
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    self.df[col] = self.df[col].clip(
                        lower=q1 - factor * (q3 - q1), upper=q3 + factor * (q3 - q1)
                    )

                case ColumnOperation.LOG_TRANSFORM:
                    if (self.df[col] <= 0).any():
                        offset = abs(self.df[col].min()) + 1
                        self.df[col] = np.log1p(self.df[col] + offset)
                    else:
                        self.df[col] = np.log(self.df[col])

                case _:  # Default case for unhandled operations
                    raise NotImplementedError(
                        f"Cleaning operation {operation.value} not implemented."
                    )

        except Exception as e:
            print(f"Error applying {operation.value} to column {col}: {e}")
