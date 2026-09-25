# ============================================================
# API — DATASETS ENDPOINTS
# GET /datasets
# GET /dataset-stats
# GET /dataset-samples
# ============================================================

import os
import json
import logging
import pandas as pd

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("drug_synergy.datasets")

router = APIRouter()


def _load_and_cache_csv(dataset_key: str) -> pd.DataFrame:
    """Load a CSV by dataset key, raise 404 if not found."""
    from app import DATASET_CONFIG
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_key}")

    csv_path = config["csv"]
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {csv_path}. "
                   "Make sure the datasets are in the data/ directory."
        )

    df = pd.read_csv(csv_path)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


# ============================================================
# GET /datasets — List all datasets with status
# ============================================================

@router.get("/datasets")
async def list_datasets():
    """
    Return all supported datasets with their training status
    (trained / not trained) and CSV availability.
    """
    from app import DATASET_CONFIG, loaded_models

    result = []
    for key, config in DATASET_CONFIG.items():
        csv_exists   = os.path.exists(config["csv"])
        model_exists = key in loaded_models

        result.append({
            "key":          key,
            "display_name": config["display_name"],
            "csv_available": csv_exists,
            "model_trained": model_exists,
            "status":        "trained" if model_exists else ("available" if csv_exists else "missing"),
        })

    return {"datasets": result}


# ============================================================
# GET /dataset-stats — Real statistics from CSV
# ============================================================

@router.get("/dataset-stats")
async def dataset_stats(dataset: str = Query(..., description="Dataset key, e.g. 'oncology'")):
    """
    Return actual statistics for the specified dataset CSV.
    All values come from the real CSV — no hardcoded numbers.
    """
    from app import DATASET_CONFIG, loaded_models

    df = _load_and_cache_csv(dataset)
    config = DATASET_CONFIG[dataset]

    # Count valid SMILES
    from ml.preprocessing import validate_smiles
    valid_drug1 = df["drugname1"].apply(lambda s: validate_smiles(str(s)) is not None).sum()
    valid_drug2 = df["drugname2"].apply(lambda s: validate_smiles(str(s)) is not None).sum()

    clean_df = df.dropna(subset=["drugname1", "drugname2", "cellline", "score"])
    score_col = clean_df["score"].dropna()

    # Score distribution bins
    import numpy as np
    counts, bin_edges = map(lambda x: x.tolist(), __import__("numpy").histogram(score_col, bins=30))

    # Get trained model metadata if available
    model_info = {}
    if dataset in loaded_models:
        model_info = loaded_models[dataset].get("metadata", {})

    return {
        "dataset":         dataset,
        "display_name":    config["display_name"],
        "total_rows":      int(len(df)),
        "valid_rows":      int(len(clean_df)),
        "missing_rows":    int(len(df) - len(clean_df)),
        "duplicate_rows":  int(df.duplicated().sum()),
        "unique_drug_pairs": int(clean_df[["drugname1", "drugname2"]].drop_duplicates().shape[0]),
        "unique_cell_lines": int(clean_df["cellline"].nunique()),
        "valid_drug1_smiles": int(valid_drug1),
        "valid_drug2_smiles": int(valid_drug2),
        "score_stats": {
            "min":    round(float(score_col.min()), 4) if len(score_col) > 0 else None,
            "max":    round(float(score_col.max()), 4) if len(score_col) > 0 else None,
            "mean":   round(float(score_col.mean()), 4) if len(score_col) > 0 else None,
            "median": round(float(score_col.median()), 4) if len(score_col) > 0 else None,
            "std":    round(float(score_col.std()), 4) if len(score_col) > 0 else None,
        },
        "score_distribution": {
            "counts":    counts,
            "bin_edges": [round(float(b), 4) for b in bin_edges],
        },
        "model_trained": dataset in loaded_models,
        "model_info":    model_info,
    }


# ============================================================
# GET /dataset-samples — Paginated table of rows
# ============================================================

@router.get("/dataset-samples")
async def dataset_samples(
    dataset: str = Query(...),
    page:    int = Query(1, ge=1),
    limit:   int = Query(50, ge=1, le=200),
    search:  str = Query("", description="Filter by cell line or drug"),
):
    """
    Return paginated rows from the dataset CSV for the explorer table.
    """
    df = _load_and_cache_csv(dataset)
    df = df.dropna(subset=["drugname1", "drugname2", "cellline", "score"])

    # Apply search filter
    if search.strip():
        mask = (
            df["cellline"].str.contains(search, case=False, na=False) |
            df["drugname1"].str.contains(search, case=False, na=False) |
            df["drugname2"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    total = len(df)
    start = (page - 1) * limit
    end   = start + limit
    page_df = df.iloc[start:end].copy()

    # Truncate SMILES for display (full SMILES are very long)
    def truncate(s, n=40):
        return s[:n] + "..." if len(str(s)) > n else s

    rows = []
    for _, row in page_df.iterrows():
        rows.append({
            "drugname1":       truncate(str(row["drugname1"])),
            "drugname2":       truncate(str(row["drugname2"])),
            "drugname1_full":  str(row["drugname1"]),
            "drugname2_full":  str(row["drugname2"]),
            "cellline":        str(row["cellline"]),
            "score":           round(float(row["score"]), 4),
        })

    return {
        "dataset":    dataset,
        "total":      total,
        "page":       page,
        "limit":      limit,
        "total_pages": (total + limit - 1) // limit,
        "rows":       rows,
    }


# ============================================================
# GET /cell-lines — Known cell lines for a trained model
# ============================================================

@router.get("/cell-lines")
async def get_cell_lines(dataset: str = Query(...)):
    """
    Return the list of cell lines known to the trained model for a dataset.
    These are the cell lines from the training data.
    """
    from app import loaded_models

    if dataset not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{dataset}' not trained yet."
        )

    encoder = loaded_models[dataset]["encoder"]
    from ml.encoding import get_known_cell_lines
    cell_lines = get_known_cell_lines(encoder)

    return {
        "dataset":    dataset,
        "cell_lines": sorted(cell_lines),
        "count":      len(cell_lines),
    }
