"""
Utility functions for data loading, splitting, feature separation,
normalisation, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve
import shap
shap.initjs()


def find_path(folder, dataname):
    """Construct a CSV file path relative to this script's directory."""
    file_path = Path(__file__).parent.joinpath(folder, dataname + '.csv')
    return str(file_path)


def save_path(folder, name, file_type):
    """Construct a save file path relative to this script's directory."""
    file_path = Path(__file__).parent.joinpath(folder, name + file_type)
    return str(file_path)


def data_split(file_path):
    """
    A 50-event gap is used between train/validation and train/test boundaries
    to prevent data leakage from the sliding window feature extraction.
    """
    geodata_raw = pd.read_csv(file_path, index_col=False)

    train, test = train_test_split(geodata_raw, test_size=0.3, shuffle=False)
    train = train[:-50]
    train_r, val = train_test_split(train, test_size=0.3, shuffle=False)
    train_r = train_r[:-50]

    return train_r, val, train, geodata_raw


def combine_all_fs(ind_data, tsf_data):
    data = pd.concat([ind_data, tsf_data], axis=1)
    actual_mag = data['Actual Mag']
    data = data.drop('Actual Mag', axis=1)
    data['Actual Mag'] = actual_mag
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)

    return data


def sep_x_y(geodata, mag):
    geodata.dropna(inplace=True)
    geodata = geodata.reset_index(drop=True)
    x, y = geodata.iloc[:, :-1], geodata.iloc[:, -1]
    y = (y >= mag).astype(int)

    return x, y


def custom_data_split(trainVal, thresh, fs_type):
    """
    Split the training+validation set and select feature subsets by type. 
    A 50-event gap is maintained between train_r and the validation portion.
    """
    train_r = trainVal.iloc[:round(trainVal.shape[0] * 0.7) - 50, :]
    X_train, y_train = sep_x_y(trainVal, thresh)
    X_train_r, y_train_r = sep_x_y(train_r, thresh)

    if fs_type == 'SSF':
        X_train = X_train.iloc[:, 0:60]
        X_train_r = X_train_r.iloc[:, 0:60]
    elif fs_type == 'ts':
        X_train = X_train.iloc[:, 60:]
        X_train_r = X_train_r.iloc[:, 60:]

    return X_train_r, y_train_r, X_train, y_train


def read_data(data_name):
    dataset = pd.read_csv(data_name)
    dataset.columns = ['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']
    dataset = dataset[['Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
    dataset.dropna(inplace=True)
    dataset['Time'] = pd.to_datetime(dataset['Time'])

    return dataset


def combine_time_col(data):
    data.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
                     'Latitude', 'Longitude', 'Depth', 'Magnitude']
    new_data = data.copy().iloc[:, 6:]
    new_data.insert(0, 'Time', pd.to_datetime(data[['Year', 'Month', 'Day',
                                                      'Hour', 'Minute', 'Second']]))
    new_data = new_data.reset_index().iloc[:, 1:]

    return new_data


def compute_class_weights(y_train):
    """Compute balanced class weights for the given target variable."""
    return compute_class_weight(class_weight="balanced",
                                classes=np.unique(y_train), y=y_train)


def shap_test_eval(model, X_test, mag, region, fs_type):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(h * 10, h)
    plt.ylabel('ylabel', fontsize=0.05)
    file_name = f"{region}_{mag}_{fs_type}"
    file_path = save_path('results', file_name, '.png')
    plt.savefig(file_path)
    plt.close()


def roc_plot(y_test, y_pro, auc, label):
    fpr, tpr, _ = roc_curve(y_test, y_pro)
    plt.plot(fpr, tpr, label=label + str(auc))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()
