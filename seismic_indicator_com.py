"""
Seismic indicator computation utilities.

The 60 seismic statistical features (SSFs) used in the paper are pre-computed
using MATLAB. This module provides:
- `background_compute`: Recomputes Beta and Z values using the training set
  as the background period (to prevent data leakage for the test set).
- `replace_ZBeta`: Replaces the Beta and Z columns in the indicator data.
- `Event_Distribution`: Computes date-wise event distribution (used by Z-value).
- `daysdif`: Date difference utility.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def daysdif(start_date, end_date):
    """
    Compute the number of days between two dates.

    Parameters
    ----------
    start_date : str
        Start date in "m/d/Y" format.
    end_date : str
        End date in "m/d/Y" format.

    Returns
    -------
    int
        Number of days between the two dates.
    """
    start = datetime.strptime(start_date, "%m/%d/%Y")
    end = datetime.strptime(end_date, "%m/%d/%Y")
    return (end - start).days


def Event_Distribution(days):
    """
    Calculate date-wise distribution of seismic events in the catalogue.

    After getting the distribution, zero-padding is applied for days without
    events. This distribution is used to calculate the variance of events,
    which feeds into the Z-value computation.

    Parameters
    ----------
    days : np.ndarray
        Array of shape (n_events, 3+) with date information.
        Column indices: [0]=year, [1]=month, [2]=day.

    Returns
    -------
    np.ndarray
        Array where each row is [year, month, day, event_count].
    """
    distribution = []
    counter = 0
    loop_count = 0
    comp = days[0, 2]

    while loop_count < days.shape[0]:
        loop_count += 1
        if comp == days[loop_count - 1, 2]:
            counter += 1
        else:
            distribution.append(np.append(days[loop_count - 2, :], counter))
            counter = 0
            comp = days[loop_count - 1, 2]
            loop_count -= 1

    distribution.append(np.append(days[loop_count - 1, :], counter))

    return np.array(distribution)


def background_compute(catalog, events_n):
    """
    Recompute Beta and Z values using only the training period as background.

    The original MATLAB-computed Beta and Z values use the entire catalogue
    as background, which causes data leakage for the test set. This function
    recalculates them using only the training+validation portion (70% minus
    50-event gap) as the background period.

    Parameters
    ----------
    catalog : pd.DataFrame
        Full earthquake catalogue with columns:
        [Year, Month, Day, ..., Magnitude] where Magnitude is at index 9.
    events_n : int
        Number of events in each sliding window (e.g., 50).

    Returns
    -------
    betas : list of float
        Recomputed Beta values for each window.
    zs : list of float
        Recomputed Z values for each window.
    """
    days = catalog.iloc[:, 0:3].values
    mag = catalog.iloc[:, 9].values

    total_events = round(len(mag) * 0.7) - 50  # training+validation set with gap to avoid data leakage

    first_date = f"{days[0, 1]}/{days[0, 2]}/{days[0, 0]}"
    last_date = f"{days[total_events-1, 1]}/{days[total_events-1, 2]}/{days[total_events-1, 0]}"
    total_duration = daysdif(first_date, last_date)
    betas = []
    zs = []
    for i in range(events_n, len(mag)):
        end_t = f"{days[i, 1]}/{days[i, 2]}/{days[i, 0]}"
        start_t = f"{days[i-events_n+1, 1]}/{days[i-events_n+1, 2]}/{days[i-events_n+1, 0]}"
        t_days = daysdif(start_t, end_t)
        delta = t_days / total_duration
        beta = (events_n - (total_events * delta)) / np.sqrt(total_events * delta * (1 - delta))
        r1 = (total_events - events_n) / (total_duration - t_days)  # Background

        if t_days == 0:
            r2 = 65535
        else:
            r2 = events_n / t_days  # Current interval

        days1 = days[:total_events, :]  # background period
        distribution1 = Event_Distribution(days1)
        days2 = days[i-events_n+1:i+1, :]  # current interval
        distribution2 = Event_Distribution(days2)

        if (total_duration - t_days - len(distribution1)) > 0:
            distribution_padded1 = np.hstack((distribution1[:, 3], np.zeros(total_duration - t_days - len(distribution1))))
        else:
            distribution_padded1 = distribution1[:, 3]
        if (t_days - len(distribution2)) > 0:
            distribution_padded2 = np.hstack((distribution2[:, 3], np.zeros(t_days - len(distribution2))))
        else:
            distribution_padded2 = distribution2[:, 3]

        s1 = np.var(distribution_padded1, ddof=1)
        s2 = np.var(distribution_padded2, ddof=1)
        z = (r1 - r2) / np.sqrt((s1 / (total_events - events_n)) + (s2 / events_n))
        betas.append(beta)
        zs.append(z)

    return betas, zs


def replace_ZBeta(cat, betas, zs):
    """
    Replace the Beta and Z value columns in the indicator dataframe.

    Parameters
    ----------
    cat : pd.DataFrame
        Indicator dataframe with 'Beta Value' and 'Z Value' columns.
    betas : list of float
        New Beta values.
    zs : list of float
        New Z values.

    Returns
    -------
    pd.DataFrame
        Updated indicator dataframe.
    """
    cat['Beta Value'] = betas
    cat['Z Value'] = zs
    return cat
