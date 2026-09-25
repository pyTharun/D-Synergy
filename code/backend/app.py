# ============================================================
# FASTAPI MAIN APPLICATION
# Drug Combination Synergy Predictor — Backend
# ============================================================

import os
import sys
import json
import logging

# Ensure Windows Conda Library DLLs are available for numpy/mkl/scipy
if os.name == "nt":
    conda_base = sys.prefix
    for sub in ["", "Scripts", r"Library\bin", r"Library\usr\bin", r"Library\mingw-w64\bin"]:
        p = os.path.join(conda_base, sub)
        if os.path.exists(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.predict  import router as predict_router
from api.datasets import router as datasets_router
from api.models   import router as models_router
from api.training import router as training_router

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drug_synergy")

# ============================================================
# DATASET CONFIGURATION
# Maps dataset keys → CSV file paths and model directories
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")    # E:\synergy\code\backend\data\
MODELS_DIR = os.path.join(BASE_DIR, "models")  # E:\synergy\code\backend\models\

# Dataset config: name → {csv path, model dir, display name}
DATASET_CONFIG = {
    "oncology": {
        "csv":          os.path.join(DATA_DIR, "OncologyScreenLINCS_PRISM.csv"),
        "model_dir":    os.path.join(MODELS_DIR, "oncology"),
        "display_name": "OncologyScreen",
    },
    "oneil": {
        "csv":          os.path.join(DATA_DIR, "OneilLINCS_PRISM.csv"),
        "model_dir":    os.path.join(MODELS_DIR, "oneil"),
        "display_name": "Oneil",
    },
    "drugcomb": {
        "csv":          os.path.join(DATA_DIR, "DrugComb_LINCS_PRISM.csv"),
        "model_dir":    os.path.join(MODELS_DIR, "drugcomb"),
        "display_name": "DrugComb",
    },
    "drugcombdb": {
        "csv":          os.path.join(DATA_DIR, "DrugCombDBLINCS_PRISM.csv"),
        "model_dir":    os.path.join(MODELS_DIR, "drugcombdb"),
        "display_name": "DrugCombDB",
    },
    "almanac": {
        "csv":          os.path.join(DATA_DIR, "AlmanacLINCS_PRISM.csv"),
        "model_dir":    os.path.join(MODELS_DIR, "almanac"),
        "display_name": "Almanac",
    },
}

# ============================================================
# APP + CORS
# ============================================================

app = FastAPI(
    title="Drug Combination Synergy Predictor API",
    description=(
        "Machine-learning based prediction of drug-combination synergy scores "
        "across cancer cell lines using Morgan fingerprints, DRS (Drug Resistance Signatures), and regression models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL MODEL STORE (loaded at startup)
# ============================================================

loaded_models: dict = {}


@app.on_event("startup")
async def load_all_models():
    """
    At startup, attempt to load all available trained models into memory.
    If a model pickle is corrupted or incompatible with the current scikit-learn
    version, automatically retrain it so all models are immediately operational.
    """
    from ml.prediction import load_model_artifacts
    from ml.training import run_training_pipeline

    for dataset_key, config in DATASET_CONFIG.items():
        model_dir = config["model_dir"]
        model_path = os.path.join(model_dir, "model.pkl")

        if os.path.exists(model_path):
            try:
                artifacts = load_model_artifacts(model_dir)
                loaded_models[dataset_key] = artifacts
                logger.info(f"✓ Loaded model for dataset: {dataset_key}")
            except Exception as e:
                logger.warning(f"✗ Model for {dataset_key} needs retraining ({e}). Retraining...")
                try:
                    run_training_pipeline(
                        dataset_path=config["csv"],
                        dataset_name=config["display_name"],
                        out_dir=config["model_dir"],
                        model_type="linear",
                    )
                    loaded_models[dataset_key] = load_model_artifacts(model_dir)
                    logger.info(f"✓ Retrained and loaded model for: {dataset_key}")
                except Exception as ex:
                    logger.error(f"✗ Failed to retrain {dataset_key}: {ex}")
        elif os.path.exists(config["csv"]):
            logger.info(f"  Training initial model for {dataset_key}...")
            try:
                run_training_pipeline(
                    dataset_path=config["csv"],
                    dataset_name=config["display_name"],
                    out_dir=config["model_dir"],
                    model_type="linear",
                )
                loaded_models[dataset_key] = load_model_artifacts(model_dir)
                logger.info(f"✓ Trained and loaded model for: {dataset_key}")
            except Exception as ex:
                logger.error(f"✗ Failed to train {dataset_key}: {ex}")

    logger.info(f"Startup complete. Loaded {len(loaded_models)} model(s).")


# ============================================================
# INCLUDE ROUTERS
# ============================================================

app.include_router(predict_router,  prefix="",         tags=["Prediction"])
app.include_router(datasets_router, prefix="",         tags=["Datasets"])
app.include_router(models_router,   prefix="",         tags=["Models"])
app.include_router(training_router, prefix="",         tags=["Training"])


# ============================================================
# HEALTH & ROOT CHECK (WEB UI)
# ============================================================

@app.get("/", response_class=FileResponse, tags=["Web UI"])
async def root():
    static_file = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {
        "message": "Drug Combination Synergy Predictor API is running",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
        "loaded_models": list(loaded_models.keys()),
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
    }


# ============================================================
# RUN (for direct execution: python app.py)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    os.chdir(BASE_DIR)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
