# ============================================================
# API — MODELS ENDPOINTS
# GET /model-metrics
# GET /model-actuals
# GET /models
# ============================================================

import os
import json
import logging

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("drug_synergy.models")

router = APIRouter()


# ============================================================
# GET /models — All trained models summary
# ============================================================

@router.get("/models")
async def list_models():
    """
    Return a summary of all trained models with their metrics.
    All values come from the actual saved metadata.json files.
    """
    from app import loaded_models, DATASET_CONFIG

    result = []
    for dataset_key, artifacts in loaded_models.items():
        meta = artifacts.get("metadata", {})
        result.append({
            "dataset":          dataset_key,
            "display_name":     DATASET_CONFIG[dataset_key]["display_name"],
            "model_name":       meta.get("model_name", "Unknown"),
            "model_type":       meta.get("model_type", "unknown"),
            "morgan_radius":    meta.get("morgan_radius", 2),
            "fingerprint_size": meta.get("fingerprint_size", 2048),
            "training_samples": meta.get("training_samples", 0),
            "testing_samples":  meta.get("testing_samples", 0),
            "trained_at":       meta.get("trained_at", ""),
            "metrics":          meta.get("metrics", {}),
        })

    return {"models": result}


# ============================================================
# GET /model-metrics — Metrics for a specific dataset model
# ============================================================

@router.get("/model-metrics")
async def model_metrics(dataset: str = Query(...)):
    """
    Return actual model metrics (MAE, MSE, RMSE, R²) for a trained model.
    Values come from the real training run — never hardcoded.
    """
    from app import loaded_models, DATASET_CONFIG

    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

    if dataset not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{dataset}' not trained yet. Use POST /train first."
        )

    meta = loaded_models[dataset].get("metadata", {})

    return {
        "dataset":          dataset,
        "display_name":     DATASET_CONFIG[dataset]["display_name"],
        "model_name":       meta.get("model_name", "Unknown"),
        "model_type":       meta.get("model_type", ""),
        "morgan_radius":    meta.get("morgan_radius", 2),
        "fingerprint_size": meta.get("fingerprint_size", 2048),
        "training_samples": meta.get("training_samples", 0),
        "testing_samples":  meta.get("testing_samples", 0),
        "trained_at":       meta.get("trained_at", ""),
        "metrics":          meta.get("metrics", {}),
        "dataset_stats":    meta.get("dataset_stats", {}),
    }


# ============================================================
# GET /model-actuals — Actual vs Predicted data for scatter plot
# ============================================================

@router.get("/model-actuals")
async def model_actuals(dataset: str = Query(...)):
    """
    Return actual vs predicted synergy score pairs for scatter plot.
    Comes from the real training evaluation run.
    """
    from app import DATASET_CONFIG, loaded_models

    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

    if dataset not in loaded_models:
        raise HTTPException(status_code=503, detail=f"Model for '{dataset}' not trained yet.")

    # Load from saved file
    config    = DATASET_CONFIG[dataset]
    model_dir = config["model_dir"]
    actuals_path = os.path.join(model_dir, "actuals.json")

    if not os.path.exists(actuals_path):
        raise HTTPException(
            status_code=404,
            detail="Actuals file not found. Re-train the model to generate it."
        )

    with open(actuals_path) as f:
        data = json.load(f)

    return {
        "dataset":   dataset,
        "actual":    data.get("actual", []),
        "predicted": data.get("predicted", []),
        "count":     len(data.get("actual", [])),
    }


# ============================================================
# GET /model-errors — Error distribution for histogram
# ============================================================

@router.get("/model-errors")
async def model_errors(dataset: str = Query(...)):
    """
    Return prediction error distribution for histogram visualization.
    """
    from app import DATASET_CONFIG, loaded_models

    if dataset not in loaded_models:
        raise HTTPException(status_code=503, detail=f"Model for '{dataset}' not trained yet.")

    config    = DATASET_CONFIG[dataset]
    model_dir = config["model_dir"]
    errors_path = os.path.join(model_dir, "errors.json")

    if not os.path.exists(errors_path):
        raise HTTPException(status_code=404, detail="Errors file not found. Re-train the model.")

    with open(errors_path) as f:
        data = json.load(f)

    return {
        "dataset":   dataset,
        "bin_edges": data.get("bin_edges", []),
        "counts":    data.get("counts", []),
    }
