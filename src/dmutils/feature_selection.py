"""
This module contains utility functions for the "Data Mining for Business
Analytics" book. The code you find here is adapted from:

    https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py
"""
import itertools

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


def backward_elimination(variables, train_model, score_model, verbose=False):
    """
    Variable selection using backward elimination.
    Function adapted from https://github.com/gedeck/dmba/blob/master/src/dmba/featureSelection.py

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model: better models have lower scores

    Returns:
        (best_model, best_variables)
    """
    # we start with a model that contains all variables
    best_variables = list(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print("Variables: " + ", ".join(variables))
        print("Start: score={:.2f}".format(best_score))

    while len(best_variables) > 1:
        step = [(best_score, None, best_model)]
        for remove_var in best_variables:
            step_var = list(best_variables)
            step_var.remove(remove_var)
            step_model = train_model(step_var)
            step_score = score_model(step_model, step_var)
            step.append((step_score, remove_var, step_model))

        # sort by ascending score
        step.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, removed_step, best_model = step[0]
        if verbose:
            print("Step: score={:.2f}, remove={}".format(best_score, removed_step))
        if removed_step is None:
            # stop here, as a removing more variables is detrimental to performance
            break
        best_variables.remove(removed_step)
    return best_model, best_variables
