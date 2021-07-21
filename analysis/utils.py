"""
this module contains some utility functions that is used throughout the book
"""
import math

from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

def regression_summary(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """Print regression performance metrics.
    Function adapted from https://github.com/gedeck/dmba/blob/master/src/dmba/metric.py

    Input:
        y_true: actual values
        y_pred: predicted values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_res = y_true - y_pred

    metrics = [
        ('Mean Error (ME)', sum(y_res) / len(y_res)),
        ('Root Mean Squared Error (RMSE)', math.sqrt(mean_squared_error(y_true, y_pred))),
        ('Mean Absolute Error (MAE)', sum(abs(y_res)) / len(y_res)),
    ]
    if all(yt != 0 for yt in y_true):
        metrics.extend([
            ('Mean Percentage Error (MPE)', 100 * sum(y_res / y_true) / len(y_res)),
            ('Mean Absolute Percentage Error (MAPE)', 100 * sum(abs(y_res / y_true) / len(y_res))),
        ])
    fmt1 = '{{:>{}}} : {{:.4f}}'.format(max(len(m[0]) for m in metrics))
    print('\nRegression statistics\n')
    for metric, value in metrics:
        print(fmt1.format(metric, value))
