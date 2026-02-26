"""
tsfresh feature extraction and imputation utilities.

This module handles:
- Creating sliding event windows for tsfresh input
- Extracting comprehensive time series features
- Imputing inf/nan values using only past data (no data leakage)
- Dropping linear trend features and high-NaN columns
"""

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


def id_create(data, event_n):
    """
    Create overlapping sliding windows of `event_n` events for tsfresh input.

    Each window is assigned a unique 'id' column for tsfresh's column_id.

    Parameters
    ----------
    data : pd.DataFrame
        Earthquake catalogue with at least 'Time' and 'Magnitude' columns.
    event_n : int
        Number of events per window (e.g., 50).

    Returns
    -------
    pd.DataFrame
        Concatenated windows with an 'id' column identifying each window.
    """
    dfs_list = []
    index = event_n  # index of the (event_n + 1)th row
    while index < len(data):
        tb = index - event_n
        df = data.iloc[tb:index]
        df['id'] = index - event_n
        dfs_list.append(df)
        index += 1
    data = pd.concat(dfs_list)

    return data


def ts_features_extract(data):
    """
    Extract comprehensive tsfresh features from windowed earthquake data.

    Note: The `impute` parameter of tsfresh is intentionally NOT used here
    as it introduces data leakage. Imputation is handled separately by
    `impute_replace()` using only past data.

    Parameters
    ----------
    data : pd.DataFrame
        Windowed data with 'Time', 'Magnitude', and 'id' columns.

    Returns
    -------
    pd.DataFrame
        Extracted features indexed by window id.
    """
    extracted_features = extract_features(
        data[['Time', 'Magnitude', 'id']],
        column_id='id',
        column_sort='Time',
        default_fc_parameters=ComprehensiveFCParameters()
    )

    return extracted_features


def subset(col, replace_val, features):
    """
    Identify indices where a column contains a specific value and prepare
    a series with those values replaced by NaN.

    Parameters
    ----------
    col : str
        Column name.
    replace_val : float
        Value to find and replace (e.g., np.inf, -np.inf).
    features : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    indics_to_check : list
        Indices where the value was found.
    series : pd.Series
        Column with found values replaced by NaN.
    """
    series = features[col]
    indics_to_check = series[series.isin([replace_val])].index.to_list()
    if replace_val != np.nan:
        series = series.replace(replace_val, np.nan)

    return indics_to_check, series


def replace(cols, replacement, features):
    """
    Replace inf/-inf/NaN values using only past data to avoid leakage.

    For each problematic value, only data from earlier indices (0:i) is
    used to compute the replacement.

    Parameters
    ----------
    cols : list of str
        Columns to process.
    replacement : str
        Strategy: 'max' (for +inf), 'min' (for -inf), or 'median' (for NaN).
    features : pd.DataFrame
        Feature dataframe (modified in place).

    Returns
    -------
    pd.DataFrame
        Features with values replaced.
    """
    for col in cols:
        if replacement == 'max':
            indics_to_check, series = subset(col, np.inf, features)
            for i in indics_to_check:
                features.loc[i, col] = series[0:i].max()
        elif replacement == 'min':
            indics_to_check, series = subset(col, -np.inf, features)
            for i in indics_to_check:
                features.loc[i, col] = series[0:i].min()
        elif replacement == 'median':
            indics_to_check, series = subset(col, np.nan, features)
            for i in indics_to_check:
                if not series[0:i].dropna().empty:  # if nan at beginning, keep nan then drop these rows in final processing
                    sub_series = np.ma.masked_invalid(series[0:i].values)
                    features.loc[i, col] = np.ma.mean(sub_series)
                    series[i] = np.ma.mean(sub_series)
    return features


def drop_linear_trend(df):
    """Remove all columns containing 'linear' in their name."""
    columns = df.columns
    columns_retain = [x for x in columns if 'linear' not in x]
    return df[columns_retain]


def impute_replace(features, percent):
    """
    Clean and impute tsfresh features without data leakage.

    Steps:
    1. Drop columns that are entirely NaN.
    2. Drop columns where NaN percentage exceeds `percent`.
    3. Replace +inf with past maximum of that column.
    4. Replace -inf with past minimum of that column.
    5. Replace NaN with past mean of that column.
    6. Drop linear trend features.

    Parameters
    ----------
    features : pd.DataFrame
        Raw tsfresh features.
    percent : float
        Maximum allowable NaN percentage per column.

    Returns
    -------
    pd.DataFrame
        Cleaned feature set.
    """
    features.dropna(axis=1, how='all', inplace=True)

    nan_col_percent = features.isna().sum() / len(features) * 100
    nan_col_drop = nan_col_percent[nan_col_percent > percent].index.tolist()
    features.drop(columns=nan_col_drop, inplace=True)

    nan_col = features.columns[features.isna().any()].to_list()
    neg_inf_col = features.columns[features.isin([-np.inf]).any()].to_list()
    inf_col = features.columns[features.isin([np.inf]).any()].to_list()

    features = replace(inf_col, 'max', features)
    features = replace(neg_inf_col, 'min', features)
    features = replace(nan_col, 'median', features)

    features = drop_linear_trend(features)

    return features
