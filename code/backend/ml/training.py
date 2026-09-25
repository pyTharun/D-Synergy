# ============================================================
# TRAINING MODULE
# Builds the feature matrix and trains regression models
# ============================================================

import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from .preprocessing import preprocess_dataset
from .fingerprints import get_morgan_generator, generate_fingerprint_matrix
from .encoding import fit_cell_line_encoder, encode_cell_lines, save_encoder
from .evaluation import evaluate_model, get_actual_vs_predicted, get_error_distribution


# ============================================================
# SUPPORTED MODELS
# ============================================================

SUPPORTED_MODELS = {
    "linear": "Linear Regression",
    "lasso": "LASSO Regression",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
}


def _create_model(model_type: str, **kwargs):
    """
    Instantiate a regression model by type name.
    """
    if model_type == "linear":
        return LinearRegression()

    elif model_type == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(
            alpha=kwargs.get("alpha", 0.05),
            random_state=kwargs.get("random_state", 42),
            max_iter=1000,
        )

    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1
        )

    elif model_type == "adaboost":
        return AdaBoostRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.5),
            random_state=kwargs.get("random_state", 42)
        )

    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 5),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1
        )

    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 5),
            random_state=kwargs.get("random_state", 42)
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: {list(SUPPORTED_MODELS.keys())}")


# ============================================================
# 1. BUILD THE FULL FEATURE MATRIX
# ============================================================

def prepare_features(df, morgan_gen, encoder=None, fp_size: int = 2048, fit_encoder: bool = True):
    """
    Build the full feature matrix from a cleaned DataFrame.

    Steps:
        1. Generate Drug 1 Morgan fingerprints  → (N, fp_size)
        2. Generate Drug 2 Morgan fingerprints  → (N, fp_size)
        3. Encode cell lines                    → (N, n_cell_lines)
        4. Concatenate all                      → (N, 2*fp_size + n_cell_lines)

    Args:
        df          : Cleaned DataFrame with drugname1, drugname2, cellline, score
        morgan_gen  : Morgan fingerprint generator
        encoder     : Fitted OneHotEncoder (if None and fit_encoder=True, one will be created)
        fp_size     : Fingerprint size (must match generator)
        fit_encoder : If True, fit encoder on this data (use True for training data only)

    Returns:
        X       : Feature matrix (numpy array)
        y       : Target values (numpy array)
        encoder : Fitted encoder (returned so it can be saved)
    """
    from .encoding import fit_cell_line_encoder, encode_cell_lines

    print(f"\n[Training] Building feature matrix from {len(df):,} samples...")

    # Drug 1 fingerprints
    print("[Training] Generating Drug 1 fingerprints...")
    drug1_fp = generate_fingerprint_matrix(df["drugname1"], morgan_gen, fp_size)

    # Drug 2 fingerprints
    print("[Training] Generating Drug 2 fingerprints...")
    drug2_fp = generate_fingerprint_matrix(df["drugname2"], morgan_gen, fp_size)

    # Cell-line encoding
    if fit_encoder or encoder is None:
        encoder = fit_cell_line_encoder(df["cellline"])
    cell_features = encode_cell_lines(df["cellline"], encoder)

    # Concatenate all features
    X = np.hstack([drug1_fp, drug2_fp, cell_features]).astype(np.float32)
    y = df["score"].values.astype(np.float32)

    print(f"[Training] Feature matrix shape: {X.shape} (samples × features)")
    print(f"  Drug 1 fingerprints : {drug1_fp.shape[1]} bits")
    print(f"  Drug 2 fingerprints : {drug2_fp.shape[1]} bits")
    print(f"  Cell-line features  : {cell_features.shape[1]} categories")

    return X, y, encoder


# ============================================================
# 2. TRAIN A REGRESSION MODEL
# ============================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "linear", **kwargs):
    """
    Train a regression model on the provided training data.

    Args:
        X_train    : Feature matrix (training split)
        y_train    : Target values (training split)
        model_type : One of 'linear', 'random_forest', 'adaboost', 'gradient_boosting'

    Returns:
        Trained model object
    """
    model = _create_model(model_type, **kwargs)
    print(f"\n[Training] Training {SUPPORTED_MODELS.get(model_type, model_type)}...")
    model.fit(X_train, y_train)
    print("[Training] Training complete.")
    return model


# ============================================================
# 3. SAVE MODEL + ENCODER + METADATA
# ============================================================

def save_model_artifacts(
    model,
    encoder,
    metrics: dict,
    dataset_stats: dict,
    out_dir: str,
    model_type: str = "linear",
    morgan_radius: int = 2,
    fp_size: int = 2048,
    n_train: int = 0,
    n_test: int = 0,
    dataset_name: str = "",
    actuals: dict = None,
    errors: dict = None,
):
    """
    Persist all model artifacts to disk:
        - model.pkl       : trained model
        - encoder.pkl     : fitted OneHotEncoder
        - metadata.json   : training info + metrics
        - actuals.json    : actual-vs-predicted data for scatter plot
        - errors.json     : error distribution for histogram
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Save encoder
    encoder_path = os.path.join(out_dir, "encoder.pkl")
    save_encoder(encoder, encoder_path)

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "model_type": model_type,
        "model_name": SUPPORTED_MODELS.get(model_type, model_type),
        "morgan_radius": morgan_radius,
        "fingerprint_size": fp_size,
        "training_samples": n_train,
        "testing_samples": n_test,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "dataset_stats": dataset_stats,
    }

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save actual-vs-predicted (for scatter plot)
    if actuals:
        actuals_path = os.path.join(out_dir, "actuals.json")
        with open(actuals_path, "w") as f:
            json.dump(actuals, f)

    # Save error distribution (for histogram)
    if errors:
        errors_path = os.path.join(out_dir, "errors.json")
        with open(errors_path, "w") as f:
            json.dump(errors, f)

    print(f"\n[Training] Artifacts saved to: {out_dir}")
    print(f"  model.pkl, encoder.pkl, metadata.json")
    return metadata


# ============================================================
# 4. FULL TRAINING PIPELINE (end-to-end)
# ============================================================

def run_training_pipeline(
    dataset_path: str,
    dataset_name: str,
    out_dir: str,
    model_type: str = "linear",
    test_size: float = 0.20,
    morgan_radius: int = 2,
    fp_size: int = 2048,
    random_state: int = 42,
):
    """
    Full end-to-end training pipeline:
        1. Preprocess dataset
        2. Build feature matrix (Morgan fingerprints + cell-line encoding)
        3. Train/test split
        4. Train regression model
        5. Evaluate (MAE, MSE, RMSE, R²)
        6. Save all artifacts

    Returns: metadata dict with metrics
    """
    print(f"\n{'='*60}")
    print(f" TRAINING PIPELINE: {dataset_name} | {model_type}")
    print(f"{'='*60}\n")

    # Step 1: Preprocess
    df, stats = preprocess_dataset(dataset_path)

    # Step 2: Build Morgan generator
    morgan_gen = get_morgan_generator(radius=morgan_radius, fp_size=fp_size)

    # Step 3: Train/test split (raw dataframe)
    from sklearn.model_selection import train_test_split as tts
    df_train, df_test = tts(df, test_size=test_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print(f"\n[Training] Split: {len(df_train):,} train / {len(df_test):,} test")

    # Encode training data (fit encoder on TRAIN ONLY — no data leakage)
    from .encoding import fit_cell_line_encoder, encode_cell_lines

    drug1_train = generate_fingerprint_matrix(df_train["drugname1"], morgan_gen, fp_size)
    drug2_train = generate_fingerprint_matrix(df_train["drugname2"], morgan_gen, fp_size)
    encoder = fit_cell_line_encoder(df_train["cellline"])
    cell_train  = encode_cell_lines(df_train["cellline"], encoder)

    X_train = np.hstack([drug1_train, drug2_train, cell_train]).astype(np.float32)
    y_train = df_train["score"].values.astype(np.float32)

    # Encode test data (use fitted encoder — unknown cell lines → zeros)
    drug1_test = generate_fingerprint_matrix(df_test["drugname1"], morgan_gen, fp_size)
    drug2_test = generate_fingerprint_matrix(df_test["drugname2"], morgan_gen, fp_size)
    cell_test  = encode_cell_lines(df_test["cellline"], encoder)

    X_test = np.hstack([drug1_test, drug2_test, cell_test]).astype(np.float32)
    y_test = df_test["score"].values.astype(np.float32)

    print(f"[Training] X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Step 4: Train model
    model = train_model(X_train, y_train, model_type=model_type, random_state=random_state)

    # Step 5: Evaluate
    from .evaluation import evaluate_model, get_actual_vs_predicted, get_error_distribution
    metrics = evaluate_model(model, X_test, y_test)
    actuals = get_actual_vs_predicted(model, X_test, y_test)
    errors  = get_error_distribution(model, X_test, y_test)

    # Step 6: Save artifacts
    metadata = save_model_artifacts(
        model=model,
        encoder=encoder,
        metrics=metrics,
        dataset_stats=stats,
        out_dir=out_dir,
        model_type=model_type,
        morgan_radius=morgan_radius,
        fp_size=fp_size,
        n_train=len(df_train),
        n_test=len(df_test),
        dataset_name=dataset_name,
        actuals=actuals,
        errors=errors,
    )

    return metadata
