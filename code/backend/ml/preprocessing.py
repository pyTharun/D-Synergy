# ============================================================
# PREPROCESSING MODULE
# Handles loading, cleaning, and validation of drug synergy datasets
# ============================================================

import pandas as pd
import numpy as np
from rdkit import Chem


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path.
    Expected columns: drugname1, drugname2, cellline, score
    """
    df = pd.read_csv(path)
    print(f"[Preprocessing] Loaded {len(df):,} rows from: {path}")
    print(f"[Preprocessing] Columns: {list(df.columns)}")
    return df


# ============================================================
# 2. INSPECT DATASET
# ============================================================

def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Print basic statistics about the dataset and return a summary dict.
    """
    print("\n--- Dataset Inspection ---")
    print(f"  Shape         : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")

    score_stats = {}
    if "score" in df.columns:
        numeric_scores = pd.to_numeric(df["score"], errors="coerce")
        score_stats = {
            "min": float(numeric_scores.min()),
            "max": float(numeric_scores.max()),
            "mean": float(numeric_scores.mean()),
            "median": float(numeric_scores.median()),
            "std": float(numeric_scores.std()),
        }
        print(f"  Score stats   : {score_stats}")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "score_stats": score_stats,
        "unique_cell_lines": int(df["cellline"].nunique()) if "cellline" in df.columns else 0,
    }


# ============================================================
# 3. CLEAN DATASET
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where drugname1, drugname2, cellline, or score are missing.
    Convert score to numeric and drop invalid scores.
    """
    original_len = len(df)

    # Drop rows missing required columns
    df = df.dropna(subset=["drugname1", "drugname2", "cellline", "score"])

    # Convert score to numeric (coerce errors → NaN, then drop)
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    print(f"[Preprocessing] After cleaning: {len(df):,} rows (removed {original_len - len(df):,})")
    return df.reset_index(drop=True)


# ============================================================
# 4. VALIDATE A SINGLE SMILES STRING
# ============================================================

def validate_smiles(smiles: str):
    """
    Try to parse a SMILES string using RDKit.
    Returns the RDKit molecule object if valid, otherwise None.
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol  # None if invalid
    except Exception:
        return None


# ============================================================
# 5. FILTER ROWS WITH INVALID SMILES
# ============================================================

def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where drugname1 or drugname2 is not a valid SMILES string.
    Reports counts of invalid rows.
    """
    total = len(df)

    drug1_valid = df["drugname1"].apply(lambda s: validate_smiles(s) is not None)
    drug2_valid = df["drugname2"].apply(lambda s: validate_smiles(s) is not None)

    invalid_drug1 = int((~drug1_valid).sum())
    invalid_drug2 = int((~drug2_valid).sum())

    df = df[drug1_valid & drug2_valid].reset_index(drop=True)

    print(f"[Preprocessing] SMILES validation:")
    print(f"  Total rows       : {total:,}")
    print(f"  Invalid Drug 1   : {invalid_drug1:,}")
    print(f"  Invalid Drug 2   : {invalid_drug2:,}")
    print(f"  Valid rows kept  : {len(df):,}")

    return df


# ============================================================
# 6. HANDLE DUPLICATES
# ============================================================

def handle_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Report and optionally remove duplicate rows.
    By default keeps the first occurrence.
    """
    n_dupes = df.duplicated().sum()
    print(f"[Preprocessing] Duplicate rows found: {n_dupes:,}")

    if n_dupes > 0:
        df = df.drop_duplicates(keep=keep).reset_index(drop=True)
        print(f"[Preprocessing] After removing duplicates: {len(df):,} rows")

    return df


# ============================================================
# 7. FULL PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset(path: str) -> tuple:
    """
    Run the full preprocessing pipeline:
      1. Load
      2. Inspect
      3. Clean (missing values, numeric score)
      4. Filter invalid SMILES
      5. Handle duplicates

    Returns: (cleaned_df, stats_dict)
    """
    df = load_dataset(path)
    stats = inspect_dataset(df)

    df = clean_dataset(df)
    df = filter_valid_smiles(df)
    df = handle_duplicates(df)

    stats["valid_rows"] = len(df)
    stats["unique_drug_pairs"] = int(
        df[["drugname1", "drugname2"]].drop_duplicates().shape[0]
    )
    stats["unique_cell_lines"] = int(df["cellline"].nunique())

    # Recalculate score stats on clean data
    stats["score_stats"] = {
        "min": float(df["score"].min()),
        "max": float(df["score"].max()),
        "mean": float(df["score"].mean()),
        "median": float(df["score"].median()),
        "std": float(df["score"].std()),
    }

    print(f"\n[Preprocessing] Final dataset: {len(df):,} rows, "
          f"{stats['unique_drug_pairs']:,} drug pairs, "
          f"{stats['unique_cell_lines']:,} cell lines")

    return df, stats
