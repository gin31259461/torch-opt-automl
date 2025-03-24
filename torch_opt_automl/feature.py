import numpy as np
import pandas as pd

time_series_format = "%Y-%m-%d"
time_series_ratio = 0.5


class FeatureTypes:
    time_series = "time_series"
    categorical = "categorical"
    numerical = "numerical"
    none = "none"


def identify_feature_types(
    df: pd.DataFrame,
    pre_identified: dict[str, FeatureTypes] | None = {},
    time_series_format=time_series_format,
    time_series_ratio=time_series_ratio,
) -> dict[str, FeatureTypes]:
    """
    Identifies the type of each feature (column) in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names
              and values are the corresponding feature types.
    """

    feature_types = {}

    for col in df.columns:
        if df[col].notna().sum() == 0:
            feature_types[col] = FeatureTypes.none
            continue

        if pre_identified and pre_identified.get(col) is not None:
            feature_types[col] = pre_identified.get(col)
            continue

        if df[col].dtype == "object":
            try:
                parsed_df = pd.DataFrame()

                try:
                    parsed_df = pd.to_datetime(
                        df[col], format=time_series_format, errors="coerce"
                    )
                except ValueError:
                    parsed_df = pd.to_datetime(df[col], format="mixed", errors="coerce")

                if (parsed_df.notna().sum() / len(parsed_df)) < time_series_ratio:
                    raise ValueError(
                        "there is not has enough datetime element in given series of dataframe"
                    )

                feature_types[col] = FeatureTypes.time_series
            except ValueError:
                feature_types[col] = FeatureTypes.categorical
        elif df[col].dtype in ["datetime64", "datetime64[ns]"]:
            feature_types[col] = FeatureTypes.time_series
        elif df[col].dtype in ["int64", "float64"]:
            feature_types[col] = FeatureTypes.numerical
        elif df[col].dtype in ["bool", "boolean"]:
            feature_types[col] = FeatureTypes.categorical
        else:
            feature_types[col] = FeatureTypes.none

    return feature_types


def transform_identified_df_features(
    df: pd.DataFrame,
    feature_types: dict[str, FeatureTypes],
    time_series_format: str = time_series_format,
):
    """Converts DataFrame column types to identified feature types.

    Args:
        df (pd.DataFrame)   : The input DataFrame.
        feature_types (dict): A dictionary where keys are column names
            and values are the corresponding feature types.

    Returns:
        pd.DataFrame: The DataFrame with converted column types.
    """

    copied_df = df.copy()

    for col, col_type in feature_types.items():
        if col_type == FeatureTypes.none:
            continue
        elif col_type == FeatureTypes.time_series:
            try:
                copied_df[col] = pd.to_datetime(df[col], format=time_series_format)
            except ValueError:
                copied_df[col] = pd.to_datetime(df[col], format="mixed")
        elif col_type == FeatureTypes.categorical:
            copied_df[col] = df[col].astype("category")

    return copied_df


def clean_identified_df_and_feature_types(
    df: pd.DataFrame, feature_types: dict[str, FeatureTypes]
):
    """Cleans the identified DataFrame and feature_types dictionary by removing columns with a 'none' feature type.

    This function iterates through the `feature_types` dictionary and identifies columns with a 'none' type.
    It then creates a new DataFrame `new_df` and a new feature types dictionary `new_feature_types`, excluding the columns with a 'none' type.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_types (dict[str, FeatureTypes]): A dictionary where keys are column names and values are the corresponding feature types.

    Returns:
        tuple[pd.DataFrame, dict[str, FeatureTypes]]: A tuple containing the cleaned DataFrame and the cleaned feature types dictionary.
    """

    new_df = pd.DataFrame()
    new_feature_types = {}

    for col, col_type in feature_types.items():
        if col_type != FeatureTypes.none:
            new_df[col] = df[col]
            new_feature_types[col] = col_type

    return new_df, new_feature_types


def data_cleaning(
    df: pd.DataFrame,
    feature_types: dict[str, FeatureTypes],
    numerical_missing_strategy: str = "mean",  # 'mean', 'median', or 'drop'
    categorical_missing_strategy: str = "mode",  # 'mode' or 'drop'
    time_series_missing_strategy: str = "ffill",  # 'ffill', 'bfill', or 'drop'
    anomaly_detection_method: str = "iqr",  # 'iqr', 'zscore', or None
    anomaly_handling: str = "clip",  # 'clip' or 'drop'
    anomaly_detection_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values, duplicate data,
    and anomaly values based on feature types.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_types (dict[str, FeatureTypes]): A dictionary mapping column names to feature types.
        numerical_missing_strategy (str, optional): Strategy for handling missing values in numerical columns. Defaults to 'mean'.
        categorical_missing_strategy (str, optional): Strategy for handling missing values in categorical columns. Defaults to 'mode'.
        time_series_missing_strategy (str, optional): Strategy for handling missing values in time series columns. Defaults to 'ffill'.
        anomaly_detection_method (str, optional): Method for detecting anomalies. Defaults to 'iqr'.
        anomaly_handling (str, optional): Strategy for handling anomalies. Defaults to 'clip'.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    copied_df = df.copy()

    # Handle Missing Values
    for col, col_type in feature_types.items():
        if col_type == FeatureTypes.numerical:
            copied_df[col] = copied_df[col].astype("float64")
            if numerical_missing_strategy == "mean":
                copied_df[col] = copied_df[col].fillna(copied_df[col].mean())
            elif numerical_missing_strategy == "median":
                copied_df[col] = copied_df[col].fillna(copied_df[col].median())
            elif numerical_missing_strategy == "drop":
                copied_df.dropna(subset=[col], inplace=True)
        elif col_type == FeatureTypes.categorical:
            if categorical_missing_strategy == "mode":
                copied_df[col] = copied_df[col].fillna(copied_df[col].mode().iloc[0])
            elif categorical_missing_strategy == "drop":
                copied_df.dropna(subset=[col], inplace=True)
        elif col_type == FeatureTypes.time_series:
            if time_series_missing_strategy == "ffill":
                copied_df[col] = copied_df[col].ffill()
            elif time_series_missing_strategy == "bfill":
                copied_df[col] = copied_df[col].bfill()
            elif time_series_missing_strategy == "drop":
                copied_df.dropna(subset=[col], inplace=True)

    # Handle Duplicate Data
    copied_df.drop_duplicates(inplace=True)

    # Handle Anomaly Values
    if anomaly_detection_method == "iqr":
        for col, col_type in feature_types.items():
            if col_type == FeatureTypes.numerical:
                # more detail refer to doc: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy-quantile

                # this is a quantile equation
                # -> (1-g)*y[j] + g*y[j+1]

                # y is a sorted array
                # j = (q*n + m - 1) // 1
                # g = (q*n + m - 1) % 1

                # np.percentile
                # -> method="weibull" means using weibull distribution q
                # -> default method is linear -> (1 - q)

                Q1, Q3 = np.percentile(copied_df[col], [25, 75], method="linear")

                IQR = Q3 - Q1

                print(Q1, Q3)

                lower_bound = Q1 - anomaly_detection_threshold * IQR
                upper_bound = Q3 + anomaly_detection_threshold * IQR

                if anomaly_handling == "clip":
                    copied_df[col] = pd.Series(copied_df[col]).clip(
                        lower=lower_bound, upper=upper_bound
                    )
                elif anomaly_handling == "drop":
                    copied_df = copied_df[
                        (copied_df[col] >= lower_bound)
                        & (copied_df[col] <= upper_bound)
                    ]
    elif anomaly_detection_method == "zscore":
        pass

    copied_df = pd.DataFrame(copied_df).set_index(pd.RangeIndex(0, len(copied_df), 1))

    return copied_df


def data_transformation():
    pass


def feature_selection():
    pass


def feature_engineering():
    pass


def dimensionality_reduction():
    pass
