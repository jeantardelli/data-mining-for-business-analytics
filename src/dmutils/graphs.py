"""
This module contains utility functions for the "Data Mining for Business
Analytics" book. The code you find here is adapted from:

    https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py
"""
import pandas as pd


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
