"""
This module contains utility functions for the "Data Mining for Business
Analytics" book. The code you find here is adapted from:

    https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py
"""
import io
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz

try:
    from IPython.display import Image
except ImportError:
    Image = None
try:
    import pydotplus
except ImportError:
    pydotplus = None


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
    ax.set_ylabel('cumulative gains')
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


def plot_decision_tree(decision_tree, feature_names=None, class_names=None, impurity=False,
                       label='root', max_depth=None, rotate=False, pdf_file=None):
    """
    Create a plot of the scikit-learn decision tree and show in the Jupyter notebook.
    Function adapted from the https://github.com/gedeck/dmba/blob/master/src/dmba/graphs.py

    Input:
        decision_tree: scikit-learn decision tree
        feature_names (optional): variable names
        class_names (optional): class names, only relevant for classification trees
        impurity (optional): show node impurity
        label (optional): only show labels at the root
        max_depth (optional): limit
        rotate (optional): rotate the layout of the graph
        pdf_file (optional): provide a pathname to create a PDF file of the graph
    """
    if pydotplus is None:
        return 'You need to install pydotplus to visualize decision trees.'
    if Image is None:
        return 'You need to install ipython to visualize decision trees.'
    if class_names is not None:
        class_names = [str(s) for s in class_names] # convert to strings
    dot_data = io.StringIO()
    export_graphviz(decision_tree, feature_names=feature_names, class_names=class_names,
                    impurity=impurity, label=label, out_file=dot_data, filled=True,
                    rounded=True, special_characters=True, max_depth=max_depth, rotate=rotate)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if pdf_file is not None:
        graph.write_pdf(str(pdf_file))
    return Image(graph.create_png())


# Taken from scikit-learn documentation

def text_decision_tree(decision_tree, indent='  ', as_ratio=True):
    """
    Create a text representation of the scikit-learn decisioin tree.
    Input:
         decision_tree: scikit-learn decision tree
         as_ratio: show the composition of the leaf nodes as ratio (default)
                   instead of counts.
         indent: indentation (default two spaces)
    """
    n_nodes = decision_tree.tree_.node_count
    children_left = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    feature = decision_tree.tree_.feature
    threshold = decision_tree.tree_.threshold
    node_value = decision_tree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)] # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((childre_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    rep = []
    for i in range(n_nodes):
        common = f'{node_depth[i] * indent}node={i}'
        if is_leaves[i]:
            value = node_value[i]
            if as_ratio:
                value = [[round(vi  / sum(v), 3) for vi in v] for v in value]
            rep.append(f'{common} leaf node: {value}')
        else:
            rule = f'{children_left[i]} if {feature[i]} <= {threshold[i]} else to node {children_right[i]}'
            rep.append(f'{common} test node: got to node {rule}')
    return '\n'.join(rep)
