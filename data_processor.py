"""
Data processing pipeline for combining seismic indicator features with
tsfresh time series features.

This module handles the full preprocessing workflow:
1. Read pre-computed seismic indicator data (from MATLAB)
2. Recompute Beta/Z values for the whole dataset (leakage-free)
3. Extract tsfresh features from magnitude time series
4. Impute missing/invalid values
5. Combine indicator + tsfresh features
6. Normalise (z-score standardisation on indicator features using training stats)
7. Save processed datasets for model training and evaluation
"""

import pandas as pd
import numpy as np
from utils import find_path, data_split, combine_time_col, combine_all_fs
from tsf_utils import id_create, ts_features_extract, impute_replace
from seismic_indicator_com import background_compute, replace_ZBeta
import logging

logging.basicConfig(filename="preprocess.log", level=logging.INFO)
logger = logging.getLogger(__name__)


# Column names for the 60 seismic statistical features
INDICATOR_COLUMNS = [
    'T_Days', 'b(MLE)', 'a(MLE)', 'b(LSQ)', 'a(LSQ)',
    'std_b(MLE)', 'std_b(LSQ)', 'T_Rec_(MLE)4.0', 'T_Rec_(MLE)4.1', 'T_Rec_(MLE)4.2',
    'T_Rec_(MLE)4.3', 'T_Rec_(MLE)4.4', 'T_Rec_(MLE)4.5', 'T_Rec_(MLE)4.6', 'T_Rec_(MLE)4.7',
    'T_Rec_(MLE)4.8', 'T_Rec_(MLE)4.9', 'T_Rec_(MLE)5.0', 'T_Rec_(MLE)5.1', 'T_Rec_(MLE)5.2',
    'T_Rec_(MLE)5.3', 'T_Rec_(MLE)5.4', 'T_Rec_(MLE)5.5', 'T_Rec_(MLE)5.6', 'T_Rec_(MLE)5.7',
    'T_Rec_(MLE)5.8', 'T_Rec_(MLE)5.9', 'T_Rec_(MLE)6.0', 'T_Rec_lsq4.0', 'T_Rec_lsq4.1',
    'T_Rec_(LSQ)4.2', 'T_Rec_(LSQ)4.3', 'T_Rec_(LSQ)4.4', 'T_Rec_(LSQ)4.5', 'T_Rec_(LSQ)4.6',
    'T_Rec_(LSQ)4.7', 'T_Rec_(LSQ)4.8', 'T_Rec_(LSQ)4.9', 'T_Rec_(LSQ)5.0', 'T_Rec_(LSQ)5.1',
    'T_Rec_(LSQ)5.2', 'T_(LSQ)_(LSQ)5.3', 'T_Rec_(LSQ)5.4', 'T_Rec_(LSQ)5.5', 'T_Rec_(LSQ)5.6',
    'T_Rec_(LSQ)5.7', 'T_Rec_(LSQ)5.8', 'T_Rec_(LSQ)5.9', 'T_Rec_(LSQ)6.0', 'Beta Value',
    'Z Value', 'M_Mag ', 'dE1/2 ', 'M_def(MLE) ', 'M_def_(LSQ) ',
    'eta ', 'eta(LSQ)', 'x6 ', 'x7_(MLE) ', 'x7_(LSQ)'
]


class DataProcessor:
    """
    Pre-process raw earthquake catalogue data, extract tsfresh features,
    and combine with seismic indicator data for model ingestion.

    Parameters
    ----------
    raw_whole : str
        Name of the raw earthquake catalogue file (without extension).
    indicator_train_val : str
        Name of the pre-computed indicator file for training+validation.
    indicator_whole : str
        Name of the pre-computed indicator file for the whole dataset.
    event_n : int
        Number of events per sliding window (e.g., 50).
    percent_drop : float
        Maximum NaN percentage threshold for dropping tsfresh columns.
    thresh : float
        Magnitude threshold for binary classification.
    save_folder : str
        Folder name for reading/saving data (relative to script directory).
    """

    def __init__(self, raw_whole, indicator_train_val, indicator_whole,
                 event_n, percent_drop, thresh, save_folder):
        self.raw_wh = raw_whole
        self.indicator_tv = indicator_train_val
        self.indicator_wh = indicator_whole
        self.event_n = event_n
        self.percent_drop = percent_drop
        self.thresh = thresh
        self.save_folder = save_folder
        self.trainVal = None
        self.whole = None

    def read_indicators(self, data_name, whole_data=None):
        """
        Read pre-computed seismic indicator data from CSV.

        For the whole dataset (not trainVal), Beta and Z values are
        recomputed using only the training period as background to
        prevent data leakage.

        Parameters
        ----------
        data_name : str
            Name of the indicator CSV file (without extension).
        whole_data : pd.DataFrame, optional
            Full catalogue (required for recomputing Beta/Z on whole data).

        Returns
        -------
        pd.DataFrame
            Indicator features with 'Actual Mag' target column.
        """
        file_path = find_path(self.save_folder, data_name)
        indicator_data = pd.read_csv(file_path, header=None, index_col=False)
        actual_mag = indicator_data.iloc[:, 63]  # last column is the target
        indicator_data = indicator_data.iloc[:, 3:63]  # first three columns are date columns
        indicator_data.columns = INDICATOR_COLUMNS

        if 'trainVal' not in data_name:
            beta, z = background_compute(whole_data, self.event_n)
            indicator_data = replace_ZBeta(indicator_data, beta, z)

        # Replace sentinel values (65535, inf, -inf) with NaN
        indicator_data[(indicator_data == 65535) | (indicator_data == np.inf) | (indicator_data == -np.inf)] = np.nan

        # Replace zeros with NaN (except T_Days where 0 is valid)
        columns_with_zeros = indicator_data.columns[(indicator_data == 0).any()]
        columns_with_zeros_list = columns_with_zeros.tolist()
        if 'T_Days' in columns_with_zeros_list:
            columns_with_zeros_list.remove('T_Days')
        for col in columns_with_zeros_list:
            indicator_data.loc[indicator_data[col] == 0, col] = np.nan

        indicator_data['Actual Mag'] = actual_mag
        return indicator_data

    def impute_na(self, df):
        """
        Impute NaN values using magnitude-aware historical mean.

        For each NaN, the replacement value is the mean of the same column
        from earlier rows where Actual Mag is within [-0.3, +0.2] of the
        current row's Actual Mag. This avoids data leakage by only using
        past data.

        Parameters
        ----------
        df : pd.DataFrame
            Indicator data with potential NaN values.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """
        nan_col = df.columns[df.isna().any()].to_list()
        logger.info(f" columns to impute{nan_col}")
        for col in nan_col:
            df_nan = df[col][df[col].isnull()]
            for index, row in df_nan.items():
                inter_df = df.loc[0:index-1, [col, 'Actual Mag']]
                actual_mag = float(df.loc[index, 'Actual Mag'])
                df.loc[index, col] = inter_df[
                    (inter_df['Actual Mag'] <= actual_mag + 0.2) &
                    (inter_df['Actual Mag'] >= actual_mag - 0.3)
                ][col].mean()

        return df

    def create_all_fs_data(self):
        """
        Run the full preprocessing pipeline.

        Steps:
        1. Split the raw catalogue into train/val/test.
        2. Extract tsfresh features for both trainVal and whole datasets.
        3. Read and clean seismic indicator features.
        4. Combine indicator + tsfresh features.
        5. Normalise using training statistics only.
        6. Save to CSV.

        Returns
        -------
        trainVal : pd.DataFrame
            Processed training+validation dataset.
        whole : pd.DataFrame
            Processed complete dataset (for final train/test evaluation).
        """
        file_path = find_path(self.save_folder, self.raw_wh)
        train_r, val, train, whole_data = data_split(file_path)
        val = combine_time_col(train)
        whole = combine_time_col(whole_data)
        print(whole.shape)

        # Create tsfresh features for whole dataset
        windowed_data = id_create(whole, self.event_n)
        extracted_features_alldata = ts_features_extract(windowed_data)
        features_alldata = impute_replace(extracted_features_alldata, self.percent_drop)
        print(features_alldata.shape)

        whole_with_test = self.read_indicators(self.indicator_wh, whole_data)
        whole_with_test = self.impute_na(whole_with_test)
        print(f'whole ind: {whole_with_test.shape}')
        logger.info(f'whole ind: {whole_with_test.shape}')

        whole_with_test = combine_all_fs(whole_with_test, features_alldata)

        # Create tsfresh features for training and validation data
        windowed_data = id_create(val, self.event_n)
        extracted_features_val = ts_features_extract(windowed_data)
        features_val = impute_replace(extracted_features_val, self.percent_drop)
        print(f"val ts: {features_val.shape}")

        val = self.read_indicators(self.indicator_tv)
        val = self.impute_na(val)
        logger.info(f'val ind: {val.shape}')

        val = combine_all_fs(val, features_val)

        self.trainVal = self.normalise(val)
        self.whole = self.normalise(whole_with_test)
        logger.info(f'final trainVal and whole shapes: {self.trainVal.shape} {self.whole.shape}')

        self.save_data()

        return self.trainVal, self.whole

    def normalise(self, data):
        """
        Z-score standardise the first 60 columns (seismic indicators) using
        only the training portion's statistics.

        The tsfresh features (columns 60+) are NOT normalised because some
        are binary (0/1) indicators.

        Parameters
        ----------
        data : pd.DataFrame
            Combined feature set.

        Returns
        -------
        pd.DataFrame
            Normalised dataset.
        """
        train = data.iloc[:round(data.shape[0] * 0.7) - 50, :]
        mean = train.iloc[:, 0:60].mean()
        stdv = train.iloc[:, 0:60].std()
        data.iloc[:, 0:60] = (data.iloc[:, 0:60] - mean) / stdv

        return data

    def save_data(self):
        """Save the processed trainVal and whole datasets to CSV."""
        save_name_trainVal = f'{self.raw_wh}_all_fs_trainVal_{self.thresh}'
        save_name_whole = f'{self.raw_wh}_all_fs_whole_{self.thresh}'

        save_path_trainVal = find_path(self.save_folder, save_name_trainVal)
        self.trainVal.to_csv(save_path_trainVal, mode='a', index=False)

        save_path_whole = find_path(self.save_folder, save_name_whole)
        self.whole.to_csv(save_path_whole, mode='a', index=False)
