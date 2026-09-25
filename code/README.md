# Drug Combination Synergy Predictor AI
### Machine Learning & Deep Learning Based Prediction of Drug Synergy in Breast Cancer
**Reference:** *Machine learning and deep learning-based prediction of drug synergy in breast cancer*, Frontiers in Pharmacology (2025)  
**Authors:** Mohadeseh Mozaffarilegha and Sajjad Gharaghani  
**Project:** Project 16  

> ⚠ **Research & Educational Platform.** Computational screening tool for oncology research. Not approved for direct clinical or medical decision making.

---

## 📁 Project Workflow Directory Structure

The project is structured according to the standard Data Science & Machine Learning engineering lifecycle:

```
E:\synergy\code\
├── docs/                                 # Stage 1: Literature & Project Specifications
│   ├── Synergistic_Drug_Combination_ML.pdf# Frontiers in Pharmacology (2025) Research Paper
│   └── project_overview.md               # Scientific background, objectives, and DRS formulation
│
├── data/                                 # Stage 2: Centralized Benchmark Cancer Datasets
│   ├── OncologyScreenLINCS_PRISM.csv     # PRISM broad oncology viability screen
│   ├── OneilLINCS_PRISM.csv              # O'Neil multi-cell line combination screens
│   ├── DrugComb_LINCS_PRISM.csv          # DrugComb multi-dose combination matrix
│   ├── DrugCombDBLINCS_PRISM.csv         # DrugCombDB curated preclinical database
│   └── AlmanacLINCS_PRISM.csv            # NCI-ALMANAC FDA-approved drug combinations
│
├── notebooks/                            # Stage 3: Interactive Prototyping & Academic Review
│   ├── drug_synergy_evaluation.ipynb     # Main Multi-Model Evaluation, Metrics (MSE, R²) & Plots
│   └── oncology_exploration.ipynb        # Exploratory Data Analysis & initial data checks
│
├── pipelines/                            # Stage 4: Standalone Scripts & Training Pipelines
│   ├── train_xgboost.py                  # Standalone XGBoost pipeline with RDKit Morgan fingerprints
│   ├── train_oncology.py                 # OncologyScreen classification/regression pipeline
│   ├── train_almanac.py                  # ALMANAC large-scale training script
│   ├── train_model.py                    # Generic CLI model trainer across datasets
│   ├── inspect_dataset.py                # Fast dataset statistics and null-checker
│   └── create_notebook.py                # Notebook generation utility
│
├── backend/                              # Stage 5: Production REST API & Web Deployment
│   ├── app.py                            # FastAPI application entrypoint (Uvicorn daemon)
│   ├── requirements.txt                  # Python dependencies
│   ├── api/                              # Route handlers
│   │   ├── predict.py                    # POST /predict (real-time multi-model inference)
│   │   ├── datasets.py                   # GET /datasets, dataset statistics
│   │   ├── models.py                     # GET /models, model metadata & actuals
│   │   └── training.py                   # POST /train asynchronous model retraining
│   ├── ml/                               # Core machine learning & cheminformatics modules
│   │   ├── preprocessing.py              # Data cleaning and SMILES validation
│   │   ├── fingerprints.py               # RDKit Morgan fingerprint extraction
│   │   ├── encoding.py                   # Cell line categorical encoders
│   │   ├── training.py                   # Model fitting engine
│   │   ├── evaluation.py                 # MAE, MSE, RMSE, R² metrics
│   │   └── prediction.py                 # Real-time multi-model ensemble inference
│   ├── models/                           # Serialized scikit-learn & XGBoost model weights
│   │   ├── oncology/                     # OncologyScreen trained model & feature mappings
│   │   ├── oneil/                        # O'Neil trained model & feature mappings
│   │   ├── drugcomb/                     # DrugComb trained model & feature mappings
│   │   ├── drugcombdb/                   # DrugCombDB trained model & feature mappings
│   │   └── almanac/                      # ALMANAC trained model & feature mappings
│   ├── data/                             # Mirrored dataset store for the backend service
│   └── static/                           # Modern Web Application UI
│       └── index.html                    # 3D DNA Helix, Ambient Aura, Presets & Benchmark Modal
│
├── environment.yml                       # Conda Environment specification (Python 3.11)
└── README.md                             # Master project documentation
```

---

## 🚀 Quick Start Guide

### 1. Activate the Conda Environment
```bash
conda activate drug-synergy
```

### 2. Launch the Web Application
```powershell
cd E:\synergy\code\backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any web browser.

### 3. Run the Evaluation Jupyter Notebook
```bash
jupyter notebook notebooks/drug_synergy_evaluation.ipynb
```
Or open [`notebooks/drug_synergy_evaluation.ipynb`](notebooks/drug_synergy_evaluation.ipynb) directly in VS Code.

### 4. Run Standalone Training Pipelines
```bash
python pipelines/train_xgboost.py
python pipelines/train_oncology.py
python pipelines/train_almanac.py
```

---

## 🔬 Scientific Summary & Validated Combinations

| Drug 1 | Drug 2 | Cell Line | Class | Synergy Score | Interpretation |
|---|---|---|---|:---:|---|
| **Anastrozole** | **Methotrexate** | MCF7 | ER+ Breast | `+12.59` | **High Synergy (Green Aura)** |
| **Anastrozole** | **Methotrexate** | T47D | ER+ Breast | `+23.87` | **High Synergy (Green Aura)** |
| **Letrozole** | **Methotrexate** | MCF7 | ER+ Breast | `+11.20` | **High Synergy (Green Aura)** |
| **Anastrozole** | **Lapatinib** | T47D | ER+ Breast | `+18.45` | **High Synergy (Green Aura)** |
| **Imatinib** | **5-Fluorouracil** | Control | General | `-4.20` | **Antagonistic (Red Aura)** |

---

## 📊 Benchmark Model Performance (Frontiers in Pharmacology 2025)

| Representation / Model | Biological Feature Space | MSE ↓ | RMSE ↓ | R² Score ↑ |
|---|---|:---:|:---:|:---:|
| **SynergyX (Deep Learning)** | Drug Resistance Signatures (DRS) | **92.16 ± 1.82** | **9.60** | **0.74** |
| **XGBoost Regressor** | DRS + Morgan Fingerprints | 144.51 ± 3.40 | 12.02 | 0.62 |
| **Random Forest** | DRS + Morgan Fingerprints | 165.65 ± 4.10 | 12.87 | 0.58 |
| **AdaBoost** | DRS + Morgan Fingerprints | 182.30 ± 4.50 | 13.50 | 0.53 |
| **LASSO (L1 Regularization)** | Sparse DRS Features | 198.40 ± 5.10 | 14.08 | 0.49 |
| **Linear Regression** | Ordinary Least Squares Baseline | 210.15 ± 5.80 | 14.50 | 0.46 |
