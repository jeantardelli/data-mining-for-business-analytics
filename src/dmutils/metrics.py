"""
This module contains utility functions for the "Data Mining for Business
Analytics" book. The code you find here is adapted from:

    https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py
"""

import math

from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, r2_score

import pandas as pd
import numpy as np


def classification_summary(y_true, y_pred, class_names=None):
    """
    Print a summary of classification performance
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/graphs.py

    Input:
        y_true: actual values
        y_pred: predicted values
        class_names (optional): list of class names
    """
    confusion_matrix_ = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('Confusion Matrix (Accuracy {:.4f})\n'.format(accuracy))

    # Pretty-print confusion matrix
    cm = confusion_matrix_

    labels = class_names
    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    # Convert the confusion matrix and labels to strings
    cm = [[str(i) for i in row] for row in cm]
    labels = [str(i) for i in labels]

    # Determine the width for the first label column and the individual cells
    prediction = 'Prediction'
    actual = 'Actual'
    label_width = max(len(s) for s in labels)
    cm_width = max(max(len(s) for row in cm for s in row), label_width) + 1
    label_width = max(label_width, len(actual))

    # Construct the format statements
    fmt1 = '{{:>{}}}'.format(label_width)
    fmt2 = '{{:>{}}}'.format(cm_width) * len(labels)

    # And print the confusion matrix
    print(fmt1.format(' ') + ' ' + prediction)
    print(fmt1.format(actual), end='')
    print(fmt2.format(*labels))

    for cls, row in zip(labels, cm):
        print(fmt1.format(cls), end='')
        print(fmt2.format(*row))


def regression_summary(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """
    Print regression performance metrics.
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


def adjusted_r2_score(y_true, y_pred, model):
    """
    Calculate the adjusted R2.
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/metric.py

    Input:
        y_true: actual values
        y_pred: predicted values
        model: preditive model
    """
    n = len(y_pred)
    p = len(model.coef_)
    if p >= n - 1:
        return 0
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def AIC_score(y_true, y_pred, model=None, df=None):
    """
    Calculate Akaike Information Criterion (AIC).
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/metric.py


    Input:
        y_true: actual values
        y_pred: predicted values
        model (optional): predictive model
        df (optional): degrees of freedom of model

    One of model of df is required
    """
    if df is None and model is None:
        raise ValueError("You need to provide either model or df")
    n = len(y_pred)
    p = len(model.coef_) + 1 if df is None else df
    resid = np.array(y_true) - np.array(y_pred)
    sse = np.sum(resid ** 2)
    constant = n + n * np.log(2 * np.pi)
    return n * math.log(sse / n) + constant + 2 * (p + 1)
