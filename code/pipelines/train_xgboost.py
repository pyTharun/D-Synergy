# ============================================================
# DRUG COMBINATION SYNERGY PREDICTION
# XGBOOST REGRESSION
# ============================================================

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

import os
DATA_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "data", "OncologyScreenLINCS_PRISM.csv"),
    os.path.join(os.path.dirname(__file__), "..", "backend", "data", "OncologyScreenLINCS_PRISM.csv"),
    "data/OncologyScreenLINCS_PRISM.csv",
    "OncologyScreenLINCS_PRISM.csv"
]
csv_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), "data/OncologyScreenLINCS_PRISM.csv")
df = pd.read_csv(csv_path)


# ------------------------------------------------------------
# 2. BASIC DATA CHECK
# ------------------------------------------------------------

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nScore statistics:")
print(df["score"].describe())


# ------------------------------------------------------------
# 3. SMILES → MOLECULAR FINGERPRINT
# ------------------------------------------------------------

def smiles_to_fingerprint(smiles, n_bits=256):

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return np.zeros(
            n_bits,
            dtype=np.float32
        )

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=n_bits
    )

    array = np.zeros(
        n_bits,
        dtype=np.float32
    )

    ConvertToNumpyArray(
        fingerprint,
        array
    )

    return array


# ------------------------------------------------------------
# 4. CREATE DRUG FEATURES
# ------------------------------------------------------------

drug1_features = np.vstack(
    df["drugname1"].apply(
        smiles_to_fingerprint
    )
)

drug2_features = np.vstack(
    df["drugname2"].apply(
        smiles_to_fingerprint
    )
)


# ------------------------------------------------------------
# 5. ENCODE CELL LINES
# ------------------------------------------------------------

cellline_features = pd.get_dummies(
    df["cellline"],
    dtype=np.float32
).values


# ------------------------------------------------------------
# 6. COMBINE FEATURES
# ------------------------------------------------------------

X = np.hstack([
    drug1_features,
    drug2_features,
    cellline_features
])

print("\nFeature matrix shape:", X.shape)


# ------------------------------------------------------------
# 7. TARGET
# ------------------------------------------------------------

y = df["score"].values


# ------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# 9. CREATE XGBOOST MODEL
# ------------------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


# ------------------------------------------------------------
# 10. TRAIN
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 11. PREDICT
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 12. EVALUATE
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)