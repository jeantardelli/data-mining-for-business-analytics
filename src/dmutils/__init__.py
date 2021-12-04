"""
This module contains utility functions for the "Data Mining for Business
Analytics" book. The code you find here is adapted from:

    https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py
"""
import os
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")

from .feature_selection import exhaustive_search, backward_elimination
from .feature_selection import forward_selection, stepwise_selection
from .metrics import regression_summary, classification_summary
from .metrics import AIC_score, adjusted_r2_score
from .graphs import lift_chart, gains_chart
from .graphs import plot_decision_tree
