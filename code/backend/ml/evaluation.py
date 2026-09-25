# ============================================================
# EVALUATION MODULE
# Computes regression metrics for the trained model
# ============================================================

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. EVALUATE MODEL — RETURNS METRICS DICT
# ============================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained regression model on test data.

    Returns a dictionary with:
        MAE  — Mean Absolute Error
        MSE  — Mean Squared Error
        RMSE — Root Mean Squared Error
        R2   — Coefficient of Determination
    """
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {
        "MAE":  round(mae,  4),
        "MSE":  round(mse,  4),
        "RMSE": round(rmse, 4),
        "R2":   round(r2,   4),
    }

    print("\n[Evaluation] ========================")
    print(f"  MAE  : {metrics['MAE']}")
    print(f"  MSE  : {metrics['MSE']}")
    print(f"  RMSE : {metrics['RMSE']}")
    print(f"  R²   : {metrics['R2']}")
    print("[Evaluation] ========================\n")

    return metrics


# ============================================================
# 2. GET ACTUAL VS PREDICTED ARRAYS (for scatter plot)
# ============================================================

def get_actual_vs_predicted(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Return actual and predicted values for scatter plot visualization.

    Returns dict with 'actual' and 'predicted' as Python lists.
    Subsamples to max 2000 points for performance.
    """
    y_pred = model.predict(X_test)

    actual    = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
    predicted = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)

    # Subsample if too many points
    max_points = 2000
    if len(actual) > max_points:
        indices = np.random.choice(len(actual), max_points, replace=False)
        actual    = [actual[i]    for i in indices]
        predicted = [predicted[i] for i in indices]

    return {
        "actual":    [round(v, 4) for v in actual],
        "predicted": [round(v, 4) for v in predicted],
    }


# ============================================================
# 3. GET ERROR DISTRIBUTION (for histogram)
# ============================================================

def get_error_distribution(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute prediction errors (actual - predicted) for histogram display.

    Returns dict with 'errors' list and 'bins' list (20 bins).
    """
    y_pred = model.predict(X_test)
    errors = (y_test - y_pred).tolist()

    errors_arr = np.array(errors)
    counts, bin_edges = np.histogram(errors_arr, bins=30)

    return {
        "errors":     [round(e, 4) for e in errors],
        "bin_edges":  [round(float(b), 4) for b in bin_edges],
        "counts":     counts.tolist(),
    }
