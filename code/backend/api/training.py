# ============================================================
# API — TRAINING ENDPOINTS
# POST /train  → start training
# GET  /train/status/{job_id}  → check progress
# ============================================================

import os
import uuid
import logging
import asyncio
import threading

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger("drug_synergy.training")

router = APIRouter()

# In-memory job store (resets on server restart)
training_jobs: dict = {}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class TrainRequest(BaseModel):
    dataset:       str   = Field(...,  description="Dataset key, e.g. 'oncology'")
    model_type:    str   = Field("linear", description="Model type: linear, random_forest, adaboost, gradient_boosting")
    test_size:     float = Field(0.20, ge=0.05, le=0.5,  description="Fraction for testing (0.05–0.50)")
    morgan_radius: int   = Field(2,    ge=1, le=4,        description="Morgan fingerprint radius")
    fp_size:       int   = Field(2048, ge=256, le=4096,   description="Fingerprint bit size")
    random_state:  int   = Field(42,                       description="Random seed for reproducibility")


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str   # "queued", "running", "completed", "failed"
    message: str
    result:  dict


# ============================================================
# BACKGROUND TRAINING FUNCTION
# ============================================================

def _run_training(job_id: str, request: TrainRequest):
    """
    Execute the training pipeline in a background thread.
    Updates training_jobs[job_id] with status and results.
    """
    from app import DATASET_CONFIG, loaded_models
    from ml.training import run_training_pipeline
    from ml.prediction import load_model_artifacts

    training_jobs[job_id]["status"]  = "running"
    training_jobs[job_id]["message"] = f"Training {request.model_type} on {request.dataset}..."

    try:
        config = DATASET_CONFIG[request.dataset]

        metadata = run_training_pipeline(
            dataset_path=config["csv"],
            dataset_name=config["display_name"],
            out_dir=config["model_dir"],
            model_type=request.model_type,
            test_size=request.test_size,
            morgan_radius=request.morgan_radius,
            fp_size=request.fp_size,
            random_state=request.random_state,
        )

        # Reload model into memory store
        artifacts = load_model_artifacts(config["model_dir"])
        loaded_models[request.dataset] = artifacts

        training_jobs[job_id]["status"]  = "completed"
        training_jobs[job_id]["message"] = "Training completed successfully."
        training_jobs[job_id]["result"]  = {
            "metrics":          metadata.get("metrics", {}),
            "training_samples": metadata.get("training_samples", 0),
            "testing_samples":  metadata.get("testing_samples", 0),
            "model_name":       metadata.get("model_name", ""),
            "trained_at":       metadata.get("trained_at", ""),
        }

        logger.info(f"Training job {job_id} completed: {metadata.get('metrics', {})}")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_jobs[job_id]["status"]  = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        training_jobs[job_id]["result"]  = {}


# ============================================================
# POST /train — Start a training job
# ============================================================

@router.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start a model training job in the background.
    Returns a job_id to poll for status.
    """
    from app import DATASET_CONFIG
    from ml.training import SUPPORTED_MODELS

    # Validate dataset
    if request.dataset not in DATASET_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset: '{request.dataset}'. "
                   f"Available: {list(DATASET_CONFIG.keys())}"
        )

    # Validate model type
    if request.model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: '{request.model_type}'. "
                   f"Available: {list(SUPPORTED_MODELS.keys())}"
        )

    # Check CSV exists
    config = DATASET_CONFIG[request.dataset]
    if not os.path.exists(config["csv"]):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset CSV not found: {config['csv']}"
        )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {
        "job_id":  job_id,
        "status":  "queued",
        "message": "Training job queued.",
        "result":  {},
    }

    # Run in background thread (training is CPU-bound, not async-friendly)
    thread = threading.Thread(
        target=_run_training,
        args=(job_id, request),
        daemon=True
    )
    thread.start()

    logger.info(f"Training job {job_id} started: {request.dataset} / {request.model_type}")

    return {
        "job_id":  job_id,
        "status":  "queued",
        "message": f"Training started for dataset '{request.dataset}' with model '{request.model_type}'.",
    }


# ============================================================
# GET /train/status/{job_id} — Check training progress
# ============================================================

@router.get("/train/status/{job_id}", response_model=JobStatusResponse)
async def training_status(job_id: str):
    """
    Poll the status of a training job by its job_id.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = training_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        result=job["result"],
    )


# ============================================================
# GET /supported-models — List available model types
# ============================================================

@router.get("/supported-models")
async def supported_models():
    from ml.training import SUPPORTED_MODELS
    return {
        "models": [
            {"key": k, "name": v}
            for k, v in SUPPORTED_MODELS.items()
        ]
    }
