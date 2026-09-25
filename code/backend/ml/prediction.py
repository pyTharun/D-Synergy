# ============================================================
# PREDICTION MODULE
# End-to-end prediction from SMILES inputs to synergy score
# ============================================================

import numpy as np
import joblib
import json
import os

from .fingerprints import get_morgan_generator, generate_morgan_fingerprint
from .encoding import encode_cell_lines, get_known_cell_lines
from .preprocessing import validate_smiles


# ============================================================
# 1. LOAD A TRAINED MODEL + ENCODER
# ============================================================

def load_model_artifacts(model_dir: str) -> dict:
    """
    Load model.pkl, encoder.pkl, and metadata.json from a model directory.

    Returns:
        dict with keys: 'model', 'encoder', 'metadata'
    """
    model_path   = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found: {encoder_path}.")

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {"model": model, "encoder": encoder, "metadata": metadata}


# ============================================================
# 2. VALIDATE PREDICTION INPUTS
# ============================================================

def validate_prediction_inputs(drug1_smiles: str, drug2_smiles: str, cellline: str, encoder) -> dict:
    """
    Validate inputs before running prediction. Fallback to first known cell line if needed.
    """
    if not drug1_smiles or not drug1_smiles.strip():
        return {"valid": False, "error": "Drug 1 SMILES is required."}
    if validate_smiles(drug1_smiles) is None:
        return {"valid": False, "error": f"Invalid SMILES for Drug 1: '{drug1_smiles[:50]}'"}

    if not drug2_smiles or not drug2_smiles.strip():
        return {"valid": False, "error": "Drug 2 SMILES is required."}
    if validate_smiles(drug2_smiles) is None:
        return {"valid": False, "error": f"Invalid SMILES for Drug 2: '{drug2_smiles[:50]}'"}

    return {"valid": True, "error": None}


def predict_synergy(
    drug1_smiles: str,
    drug2_smiles: str,
    cellline: str,
    model,
    encoder,
    morgan_radius: int = 2,
    fp_size: int = 2048,
) -> float:
    """
    Predict synergy score using Morgan Fingerprints and encoded cell lines.
    """
    morgan_gen = get_morgan_generator(radius=morgan_radius, fp_size=fp_size)

    fp1 = generate_morgan_fingerprint(drug1_smiles, morgan_gen, fp_size).reshape(1, -1)
    fp2 = generate_morgan_fingerprint(drug2_smiles, morgan_gen, fp_size).reshape(1, -1)

    known_cell_lines = get_known_cell_lines(encoder)
    if not cellline or cellline not in known_cell_lines:
        cellline = known_cell_lines[0] if known_cell_lines else "DEFAULT"

    cell = encode_cell_lines([cellline], encoder)

    X = np.hstack([fp1, fp2, cell]).astype(np.float32)
    score = float(model.predict(X)[0])
    return round(score, 4)
