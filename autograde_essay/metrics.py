import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    explained_variance_score,
    mean_squared_error,
)


def calc_metrics(true: np.array, pred: np.array) -> dict:
    """Calculates all necesserary metrics

    Args:
        true (np.array): True ansers
        pred (np.array): Predicted answers

    Returns:
        dict: Dictionary with metrics
    """
    # The mean squared error
    mse = mean_squared_error(true, pred)

    # Explained variance score
    exp_var = explained_variance_score(true, pred)

    # Kappa score
    kappa = cohen_kappa_score(true, np.around(pred), weights="quadratic")

    return {"mse": mse, "explained_variance": exp_var, "kappa": kappa}
