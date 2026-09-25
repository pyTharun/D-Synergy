# ============================================================
# ENCODING MODULE
# Handles OneHotEncoding of cancer cell lines
# ============================================================

import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. FIT ENCODER ON TRAINING CELL LINES ONLY
# ============================================================

def fit_cell_line_encoder(cell_lines) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on the training set cell lines.
    Must ONLY be called on training data to avoid data leakage.

    Args:
        cell_lines : 1D array-like of cell line strings from training set

    Returns:
        Fitted OneHotEncoder
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore",   # unknown cell lines → zero vector at prediction
        sparse_output=False        # return dense numpy array
    )
    encoder.fit(np.array(cell_lines).reshape(-1, 1))

    n_categories = len(encoder.categories_[0])
    print(f"[Encoding] OneHotEncoder fitted on {n_categories} unique cell lines.")
    return encoder


# ============================================================
# 2. ENCODE CELL LINES USING A FITTED ENCODER
# ============================================================

def encode_cell_lines(cell_lines, encoder: OneHotEncoder) -> np.ndarray:
    """
    Transform cell line strings into one-hot encoded matrix.

    Args:
        cell_lines : 1D array-like of cell line strings
        encoder    : Fitted OneHotEncoder

    Returns:
        numpy array of shape (n_samples, n_unique_cell_lines)
    """
    matrix = encoder.transform(np.array(cell_lines).reshape(-1, 1))
    print(f"[Encoding] Cell-line matrix shape: {matrix.shape}")
    return matrix.astype(np.float32)


# ============================================================
# 3. GET KNOWN CELL LINES FROM ENCODER
# ============================================================

def get_known_cell_lines(encoder: OneHotEncoder) -> list:
    """
    Return the list of cell lines the encoder was trained on.
    """
    return list(encoder.categories_[0])


# ============================================================
# 4. SAVE / LOAD ENCODER
# ============================================================

def save_encoder(encoder: OneHotEncoder, path: str):
    """Save a fitted encoder to disk using joblib."""
    joblib.dump(encoder, path)
    print(f"[Encoding] Encoder saved to: {path}")


def load_encoder(path: str) -> OneHotEncoder:
    """Load a fitted encoder from disk."""
    encoder = joblib.load(path)
    print(f"[Encoding] Encoder loaded from: {path}")
    return encoder
