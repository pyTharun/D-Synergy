# Project 16: Machine Learning and Deep Learning-Based Prediction of Drug Synergy in Breast Cancer

## 1. Overview & Research Paper Reference
- **Title:** *Machine learning and deep learning-based prediction of drug synergy in breast cancer*
- **Journal:** *Frontiers in Pharmacology (2025)*
- **Authors:** Mohadeseh Mozaffarilegha and Sajjad Gharaghani
- **Core Novelty:** Utilizing **Drug Resistance Signatures (DRS)**—derived from transcriptomic variations between resistant and sensitive cancer cell lines ($\Delta_i^{SRD} = \mu_i^R - \mu_i^S$)—to predict drug combination synergy in breast cancer.

---

## 2. Core Problem Statement & Research Objectives
1. **The Biological Bottleneck:** Cancer heterogeneity and severe drug resistance limit monotherapies. Combination drug therapy is promising, but the combinatorial space of pairs across cell lines is too vast for wet-lab testing alone.
2. **Computational Limitation:** Traditional models rely solely on chemical structures or conventional drug signatures, failing to capture cell-specific resistance mechanisms.
3. **The Solution:** Construct DRS feature representations, benchmark ML (LASSO, Random Forest, AdaBoost, XGBoost) and DL (SynergyX) algorithms across five cancer datasets, and validate combinations for estrogen receptor-positive (ER+) breast cancer (MCF-7, T47D).

---

## 3. Five Benchmark Datasets
| Dataset | Description | Primary Metric |
|---|---|---|
| **OncologyScreen (PRISM)** | High-throughput viability screens across diverse oncology cell lines | Viability / Growth Inhibition |
| **O'Neil** | 23,000+ experimental combinations across 39 cancer cell lines | Loewe Additivity Index |
| **DrugComb** | Multi-dose combination matrix across hundreds of cancer cell lines | ZIP Synergy Score |
| **DrugCombDB** | Curated clinical and preclinical synergy database | Synergy Score Index |
| **NCI-ALMANAC** | Large-scale screen of FDA-approved oncology pairs | Growth Inhibition Synergy Score |

---

## 4. Benchmark Models & Performance Summary
| Algorithm | Mechanism | MSE (DRS) ↓ | RMSE ↓ | R² Score ↑ |
|---|---|:---:|:---:|:---:|
| **SynergyX (Deep Learning)** | Multi-layer Dense Neural Network | **92.16 ± 1.82** | **9.60** | **0.74** |
| **XGBoost Regressor** | Regularized Gradient Tree Boosting | 144.51 ± 3.40 | 12.02 | 0.62 |
| **Random Forest** | Ensemble Bagging Decision Trees | 165.65 ± 4.10 | 12.87 | 0.58 |
| **AdaBoost Regressor** | Sequential Adaptive Boosting | 182.30 ± 4.50 | 13.50 | 0.53 |
| **LASSO Regressor** | L1 Sparse Regularization | 198.40 ± 5.10 | 14.08 | 0.49 |
| **Linear Regression** | Ordinary Least Squares Baseline | 210.15 ± 5.80 | 14.50 | 0.46 |

---

## 5. Validated Biological Pairs (Breast Cancer ER+)
- **Anastrozole + Methotrexate:** MCF7 Synergy: `+12.59`, T47D Synergy: `+23.87` (Strong Synergy)
- **Letrozole + Methotrexate:** MCF7 Synergy: `+11.20` (Strong Synergy)
- **Anastrozole + Lapatinib:** T47D Synergy: `+18.45` (Strong Synergy)
- **Imatinib + 5-Fluorouracil (Negative Control):** Antagonistic: `-4.20`
