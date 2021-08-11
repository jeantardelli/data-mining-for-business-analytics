"""
this module contains some utility functions that is used throughout the book
"""
import itertools
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

def gains_chart(gains, color='C0', label=None, ax=None, figsize=None):
    """ 
    Create a gains chart using predicted values. 
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/graphs.py

    Input:
        gains: must be sorted by probability
        color (optional): color of graph
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    n_total = len(gains)  # number of records
    n_actual = gains.sum()  # number of desired records

    # get cumulative sum of gains and convert to percentage
    cum_gains = pd.concat([pd.Series([0]), gains.cumsum()])  # Note the additional 0 at the front
    gains_df = pd.DataFrame({'records': list(range(len(gains) + 1)), 'cum_gains': cum_gains})

    ax = gains_df.plot(x='records', y='cum_gains', color=color, label=label, legend=False,
                       ax=ax, figsize=figsize)

    # Add line for random gain
    ax.plot([0, n_total], [0, n_actual], linestyle='--', color='k')
    ax.set_xlabel('# records')
    ax.set_ylabel('# cumulative gains')
    return ax

def lift_chart(predicted, title='Decile Lift Chart', label_bars=True, ax=None, figsize=None):
    """ 
    Create a lift chart using predicted values
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/graphs.py

    Input:
        predictions: must be sorted by probability
        ax (optional): axis for matplotlib graph
        title (optional): set to None to suppress title
        labelBars (optional): set to False to avoid mean response labels on bar chart
    """
    # group the sorted predictions into 10 roughly equal groups and calculate the mean
    groups = [int(10 * i / len(predicted)) for i in range(len(predicted))]
    mean_percentile = predicted.groupby(groups).mean()
    # divide by the mean prediction to get the mean response
    mean_response = mean_percentile / predicted.mean()
    mean_response.index = (mean_response.index + 1) * 10

    ax = mean_response.plot.bar(color='C0', ax=ax, figsize=figsize)
    ax.set_ylim(0, 1.12 * mean_response.max() if label_bars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)

    if label_bars:
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height() + 0.1))
    return ax


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

def exhaustive_search(variables, train_model, score_model):
    """
    Variable selection using backward elimination.
    Function adapted from https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py

    Input:
         variables: complete list of variables to consider in model building
         train_model: function that returns a fitted model for a give set of variables
         score_model: function that retuns the score of a model: better models have lower scores

    Returns:
        List of best subset models for increasing number of variables
    """
    # create models of increasing size and determine the best models in each case
    result = []
    for nvariables in range(1, len(variables) + 1):
        best_subset = None
        best_score = None
        best_model = None
        for subset in itertools.combinations(variables, nvariables):
            subset = list(subset)
            subset_model = train_model(subset)
            subset_score = score_model(subset_model, subset)
            if best_subset is None or best_score > subset_score:
                best_subset = subset
                best_score = subset_score
                best_model = subset_model
        result.append({
            "n": nvariables,
            "variables": best_subset,
            "score": best_score,
            "model": best_model
        })
    return result
