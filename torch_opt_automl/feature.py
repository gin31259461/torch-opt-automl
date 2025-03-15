import pandas as pd


class FeatureTypes:
    time_series = "time_series"
    categorical = "categorical"
    numerical = "numerical"
    none = "none"


def identify_feature_types(
    df: pd.DataFrame,
    pre_identified: dict[str, FeatureTypes] | None = {},
    time_series_format="%Y-%m-%d",
    time_series_ratio=0.5,
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
                parsed_df = pd.to_datetime(
                    df[col], format=time_series_format, errors="coerce"
                )

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


def convert_df_col_types(df: pd.DataFrame, feature_types: dict[str, FeatureTypes]):
    """Converts DataFrame column types to identified feature types.

    Args:
        df (pd.DataFrame)   : The input DataFrame.
        feature_types (dict): A dictionary where keys are column names
            and values are the corresponding feature types.

    Returns:
        pd.DataFrame: The DataFrame with converted column types.
    """

    df = df.copy()

    for col, col_type in feature_types.items():
        if col_type == FeatureTypes.time_series:
            df[col] = pd.to_datetime(df[col])
        elif col_type == FeatureTypes.categorical:
            df[col] = df[col].astype("category")

    return df
