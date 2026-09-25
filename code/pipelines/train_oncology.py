# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 2. LOAD YOUR DATASET
# ============================================

import os
DATA_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "data", "OncologyScreenLINCS_PRISM.csv"),
    os.path.join(os.path.dirname(__file__), "..", "backend", "data", "OncologyScreenLINCS_PRISM.csv"),
    "data/OncologyScreenLINCS_PRISM.csv",
    "OncologyScreenLINCS_PRISM.csv"
]
csv_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), "data/OncologyScreenLINCS_PRISM.csv")
df = pd.read_csv(csv_path)


# ============================================
# 3. LOOK AT THE DATA
# ============================================

print(df.head())

print(df.shape)

print(df.columns)

print(df.isnull().sum())


# ============================================
# 4. MAKE A TINY DATASET
# ============================================

small_df = df.head(100).copy()


# ============================================
# 5. CREATE A BINARY TARGET
# ============================================

median_score = small_df["score"].median()

small_df["synergy_label"] = (
    small_df["score"] > median_score
).astype(int)


# ============================================
# 6. SELECT INPUT FEATURES
# ============================================

X = small_df[
    ["drugname1", "drugname2", "cellline"]
]

y = small_df["synergy_label"]


# ============================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# 8. TELL PYTHON WHICH COLUMNS ARE CATEGORICAL
# ============================================

categorical_features = [
    "drugname1",
    "drugname2",
    "cellline"
]


# ============================================
# 9. CONVERT TEXT INTO NUMBERS
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================
# 10. CREATE LOGISTIC REGRESSION MODEL
# ============================================

model = LogisticRegression(
    max_iter=1000
)


# ============================================
# 11. CONNECT PREPROCESSING + MODEL
# ============================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================
# 12. TRAIN THE MODEL
# ============================================

pipeline.fit(
    X_train,
    y_train
)


# ============================================
# 13. MAKE PREDICTIONS
# ============================================

y_pred = pipeline.predict(X_test)


# ============================================
# 14. EVALUATE THE MODEL
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
        y_test,
        y_pred
    )
