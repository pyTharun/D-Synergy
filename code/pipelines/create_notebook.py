import os
import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Machine Learning and Deep Learning Based Prediction of Drug Synergy in Breast Cancer\n",
    "### Project 16 • Frontiers in Pharmacology (2025) Experimental Evaluation\n",
    "**Reference:** Mohadeseh Mozaffarilegha and Sajjad Gharaghani (*Frontiers in Pharmacology, 2025*)\n",
    "\n",
    "---\n",
    "### Core Objectives:\n",
    "1. **Feature Engineering**: Molecular fingerprints (RDKit Morgan radius=2) + Cell line genomic context.\n",
    "2. **Benchmarking ML Algorithms**: Linear Regression, LASSO, Random Forest, AdaBoost, and XGBoost.\n",
    "3. **Evaluation Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.\n",
    "4. **Biological Validation**: Synergy prediction on breast cancer cell lines (MCF-7 and T47D)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Imports and Environment Setup\n",
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from rdkit import Chem, RDLogger\n",
    "RDLogger.DisableLog('rdApp.*')\n",
    "from rdkit.Chem import AllChem\n",
    "from rdkit.DataStructs import ConvertToNumpyArray\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression, Lasso\n",
    "from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "print(\"All ML and Cheminformatics libraries successfully imported!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Load PRISM / OncologyScreen Dataset\n",
    "import os\n",
    "DATA_PATHS = [\n",
    "    '../data/OncologyScreenLINCS_PRISM.csv',\n",
    "    'data/OncologyScreenLINCS_PRISM.csv',\n",
    "    '../backend/data/OncologyScreenLINCS_PRISM.csv',\n",
    "    'OncologyScreenLINCS_PRISM.csv'\n",
    "]\n",
    "csv_path = next((p for p in DATA_PATHS if os.path.exists(p)), '../data/OncologyScreenLINCS_PRISM.csv')\n",
    "df = pd.read_csv(csv_path)\n",
    "print(f\"Loaded dataset from: {csv_path}\")\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "print(\"\\nSummary Statistics:\")\n",
    "print(df.describe())\n",
    "print(\"\\nMissing values:\", df.isnull().sum().to_dict())\n",
    "print(\"Unique Cell Lines:\", df[\"cellline\"].nunique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Molecular Fingerprinting (RDKit Morgan Fingerprint)\n",
    "def smiles_to_fingerprint(smiles, n_bits=256):\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    if mol is None:\n",
    "        return np.zeros(n_bits, dtype=np.float32)\n",
    "    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)\n",
    "    arr = np.zeros(n_bits, dtype=np.float32)\n",
    "    ConvertToNumpyArray(fp, arr)\n",
    "    return arr\n",
    "\n",
    "print(\"Generating drug molecular descriptors...\")\n",
    "drug1_features = np.vstack(df[\"drugname1\"].apply(smiles_to_fingerprint))\n",
    "drug2_features = np.vstack(df[\"drugname2\"].apply(smiles_to_fingerprint))\n",
    "cellline_features = pd.get_dummies(df[\"cellline\"], dtype=np.float32).values\n",
    "\n",
    "X = np.hstack([drug1_features, drug2_features, cellline_features])\n",
    "y = df[\"score\"].values\n",
    "print(\"Feature Matrix X Shape:\", X.shape)\n",
    "print(\"Target Synergy Score y Shape:\", y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Train / Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
    "print(f\"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Train Benchmark Machine Learning Models (From Research Paper)\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"LASSO (L1)\": Lasso(alpha=0.01, random_state=42),\n",
    "    \"Random Forest\": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),\n",
    "    \"AdaBoost\": AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42),\n",
    "    \"XGBoost\": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)\n",
    "}\n",
    "\n",
    "results = []\n",
    "for name, model in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    \n",
    "    results.append({\n",
    "        \"Model\": name,\n",
    "        \"MAE\": round(mae, 4),\n",
    "        \"MSE\": round(mse, 4),\n",
    "        \"RMSE\": round(rmse, 4),\n",
    "        \"R2 Score\": round(r2, 4)\n",
    "    })\n",
    "\n",
    "results_df = pd.DataFrame(results)\n",
    "print(\"\\n=== BENCHMARK EVALUATION RESULTS ===\")\n",
    "print(results_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Evaluation Comparison Visualization\n",
    "plt.figure(figsize=(10, 4.5))\n",
    "\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.bar(results_df[\"Model\"], results_df[\"MSE\"], color=[\"#3b82f6\", \"#6366f1\", \"#8b5cf6\", \"#ec4899\", \"#10b981\"])\n",
    "plt.title(\"Mean Squared Error (Lower is Better)\", fontweight=\"bold\")\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.ylabel(\"MSE\")\n",
    "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.bar(results_df[\"Model\"], results_df[\"R2 Score\"], color=[\"#3b82f6\", \"#6366f1\", \"#8b5cf6\", \"#ec4899\", \"#10b981\"])\n",
    "plt.title(\"R2 Score (Higher is Better)\", fontweight=\"bold\")\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.ylabel(\"R2\")\n",
    "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. Biological Validation on Breast Cancer Presets (Paper Validation)\n",
    "print(\"\\n--- Biological Validation on Breast Cancer Combinations ---\")\n",
    "pairs = [\n",
    "    (\"Anastrozole + Methotrexate\", \"CC(C)(C#N)C1=CC(=CC(=C1)CN2C=NC=N2)C(C)(C)C#N\", \"CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O\", \"MCF7\"),\n",
    "    (\"Imatinib + 5-Fluorouracil (Control)\", \"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5\", \"O=C1NC(=O)NC=C1F\", \"ACH-000788\")\n",
    "]\n",
    "\n",
    "best_model = models[\"XGBoost\"]\n",
    "for name, d1, d2, cell in pairs:\n",
    "    f1 = smiles_to_fingerprint(d1)\n",
    "    f2 = smiles_to_fingerprint(d2)\n",
    "    cell_enc = np.zeros(cellline_features.shape[1], dtype=np.float32)\n",
    "    feat = np.hstack([f1, f2, cell_enc]).reshape(1, -1)\n",
    "    pred = best_model.predict(feat)[0]\n",
    "    status = \"High Synergy (Green Aura)\" if pred > 5 else (\"Synergistic\" if pred > 0 else \"Antagonistic (Red Aura)\")\n",
    "    print(f\"{name} on {cell} -> Predicted Raw Score: {pred:.3f} | Interpretation: {status}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "drug-synergy",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Determine base dir
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(base_dir, "notebooks", "drug_synergy_evaluation.ipynb")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook successfully written to: {notebook_path}")
