#!/usr/bin/env python
# ============================================================
# TRAIN MODEL — Command Line Script
# Usage: python train_model.py --dataset oncology --model linear
# ============================================================

import argparse
import os
import sys

# Add backend to path
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "..", "backend")
DATA_DIR    = os.path.join(BACKEND_DIR, "data")    # backend/data/ (where CSVs were copied)
MODELS_DIR  = os.path.join(BACKEND_DIR, "models") # backend/models/
sys.path.insert(0, BACKEND_DIR)

from ml.training import run_training_pipeline, SUPPORTED_MODELS


# Dataset configuration — maps key → CSV path and output dir
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


def main():
    parser = argparse.ArgumentParser(
        description="Train a drug synergy prediction model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset to train on (e.g. oncology)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model type (default: linear)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction (default: 0.20)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2)",
    )
    parser.add_argument(
        "--fp-size",
        type=int,
        default=2048,
        help="Fingerprint bit size (default: 2048)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]

    if not os.path.exists(config["csv"]):
        print(f"ERROR: CSV not found: {config['csv']}")
        print("Please copy the dataset CSV files to the data/ directory.")
        sys.exit(1)

    metadata = run_training_pipeline(
        dataset_path=config["csv"],
        dataset_name=config["display_name"],
        out_dir=config["model_dir"],
        model_type=args.model,
        test_size=args.test_size,
        morgan_radius=args.radius,
        fp_size=args.fp_size,
        random_state=args.random_state,
    )

    print("\n" + "="*60)
    print(" TRAINING COMPLETE")
    print("="*60)
    print(f"  Dataset  : {config['display_name']}")
    print(f"  Model    : {metadata['model_name']}")
    print(f"  MAE      : {metadata['metrics']['MAE']}")
    print(f"  RMSE     : {metadata['metrics']['RMSE']}")
    print(f"  R²       : {metadata['metrics']['R2']}")
    print(f"  Saved to : {config['model_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
