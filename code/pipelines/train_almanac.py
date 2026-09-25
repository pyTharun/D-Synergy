# ============================================
# ADA BOOST DRUG SYNERGY MODEL
# Using AlmanacLINCS_PRISM.csv
# ============================================

# Step 1: Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================
# Step 2: Load the dataset
# ============================================

import os
DATA_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "data", "AlmanacLINCS_PRISM.csv"),
    os.path.join(os.path.dirname(__file__), "..", "backend", "data", "AlmanacLINCS_PRISM.csv"),
    "data/AlmanacLINCS_PRISM.csv",
    "AlmanacLINCS_PRISM.csv"
]
csv_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), "data/AlmanacLINCS_PRISM.csv")
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print(df.head())


# ============================================
# Step 3: Check the dataset
# ============================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nColumn names:")
print(df.columns)


# ============================================
# Step 4: Create our target variable
# ============================================

# score > 0  --> synergy = 1
# score <= 0 --> synergy = 0

df["synergy"] = (df["score"] > 0).astype(int)

print("\nSynergy distribution:")
print(df["synergy"].value_counts())

print("\nSynergy percentage:")
print(df["synergy"].value_counts(normalize=True) * 100)


# ============================================
# Step 5: Select input features
# ============================================

X = df[
    [
        "drugname1",
        "drugname2",
        "cellline"
    ]
]

y = df["synergy"]


# ============================================
# Step 6: Split into training and testing data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# Step 7: Convert categorical data into numbers
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "drugname1",
                "drugname2",
                "cellline"
            ]
        )
    ]
)


# ============================================
# Step 8: Create the weak learner
# ============================================

base_model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)


# ============================================
# Step 9: Create AdaBoost
# ============================================

ada_model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=300,
    learning_rate=0.5,
    random_state=42
)


# ============================================
# Step 10: Create complete pipeline
# ============================================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("adaboost", ada_model)
    ]
)


# ============================================
# Step 11: Train the model
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================
# Step 12: Make predictions
# ============================================

y_pred = model.predict(X_test)

# Probability that sample belongs to synergy class
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================
# Step 13: Evaluate the model
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_probability))