# ============================================================
# INSPECT DATASET — Quick EDA Script
# Usage: python inspect_dataset.py --dataset oncology
# ============================================================

import argparse
import os
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "..", "backend")
sys.path.insert(0, BACKEND_DIR)

DATASET_PATHS = {
    "oncology":   os.path.join(SCRIPT_DIR, "..", "data", "OncologyScreenLINCS_PRISM.csv"),
    "oneil":      os.path.join(SCRIPT_DIR, "..", "data", "OneilLINCS_PRISM.csv"),
    "drugcomb":   os.path.join(SCRIPT_DIR, "..", "data", "DrugComb_LINCS_PRISM.csv"),
    "drugcombdb": os.path.join(SCRIPT_DIR, "..", "data", "DrugCombDBLINCS_PRISM.csv"),
    "almanac":    os.path.join(SCRIPT_DIR, "..", "data", "AlmanacLINCS_PRISM.csv"),
}

def main():
    parser = argparse.ArgumentParser(description="Inspect a drug synergy dataset.")
    parser.add_argument("--dataset", choices=list(DATASET_PATHS.keys()), required=True)
    args = parser.parse_args()

    path = DATASET_PATHS[args.dataset]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Copy the CSV files to the data/ directory.")
        sys.exit(1)

    from ml.preprocessing import preprocess_dataset
    df, stats = preprocess_dataset(path)

    print("\n" + "="*50)
    print(" FINAL STATS")
    print("="*50)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
