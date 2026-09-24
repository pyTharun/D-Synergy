# Drug-Synergy Prediction

## Improving Synergistic Drug Combination Prediction with Signature-Based Gene Expression Features in Oncology

Drug-Synergy is a machine-learning-based system designed to predict the potential synergistic effects of combining two anticancer drugs using drug-related features, cancer cell-line information, and gene expression signatures.

The project aims to support computational analysis of drug combinations by learning patterns from previously observed drug-response data and generating a predicted synergy score for a new drug pair.

---

## 📌 Problem Statement

Finding effective combinations of anticancer drugs is an important challenge in oncology. Testing a large number of possible drug combinations experimentally can be expensive and time-consuming.

This project explores a computational approach for predicting drug combination synergy using:

* Drug 1
* Drug 2
* Cancer cell line
* Gene expression signatures
* Molecular/chemical features
* Machine learning models

The objective is to develop a system that can learn from existing drug-combination data and predict whether a new combination is likely to show synergistic, additive, or antagonistic behavior.

---

## 🎯 Objectives

* Analyze drug combination and cell-line response data.
* Use gene expression signatures as predictive features.
* Extract meaningful features from drug and biological data.
* Train machine learning models for synergy prediction.
* Evaluate model performance using appropriate metrics.
* Develop a web-based interface for making predictions.
* Connect the frontend with a Python backend.
* Provide an extensible architecture for future explainable and AI-assisted analysis.

---

## 🧬 System Workflow

```text
Drug 1 ───────┐
              │
Drug 2 ───────┤
              │
Cell Line ────┤
              ▼
Gene Expression
Signature ────┤
              │
              ▼
      Feature Processing
              │
              ▼
     Trained ML Model
              │
              ▼
      Synergy Prediction
              │
              ▼
   Synergy Score / Class
```

---

## 🧠 Machine Learning

The project uses machine learning techniques to learn relationships between biological and drug-related features and observed drug synergy.

Potential models used in the project include:

* XGBoost
* Scikit-learn models
* TensorFlow-based deep learning models

The final model and preprocessing pipeline are based on the trained experimental data used in the project.

### Input Features

Depending on the trained model and dataset, features may include:

* Drug identity
* Drug molecular/chemical representation
* Cell-line information
* Gene expression features
* Drug-pair characteristics
* Other experimentally derived features

### Output

The system produces a predicted drug-combination synergy result.

The exact interpretation of the output depends on the synergy metric and labels used during model training.

---

## 💻 Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow

### Development Tools

* Visual Studio Code
* Jupyter Notebook / Google Colab
* Git
* GitHub

---

## 📂 Project Structure

```text
Drug-Synergy/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── models/
│   ├── services/
│   └── preprocessing/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── screenshots/
│
├── README.md
└── .gitignore
```

The exact contents of the folders may evolve as development progresses.

---

## 🔬 Data Processing

The project works with cleaned and prepared drug-combination and biological datasets.

The general preprocessing workflow includes:

1. Loading the dataset.
2. Cleaning missing or inconsistent values.
3. Processing drug-related information.
4. Processing cell-line information.
5. Preparing gene expression features.
6. Encoding categorical variables where required.
7. Creating the final feature matrix.
8. Splitting the data for training and evaluation.

All transformations used for prediction should remain consistent with those used during model training.

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/sathvika-panuganti/Drug-Synergy.git
cd Drug-Synergy
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the backend

```bash
cd backend
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

### 5. Open the frontend

Open:

```text
frontend/index.html
```

in a browser, or use the VS Code Live Server extension.

---

## 🔗 API

The backend exposes an API endpoint for synergy prediction.

### Health Check

```text
GET /health
```

### Prediction

```text
POST /predict
```

Example request structure:

```json
{
  "drug1": "Drug A",
  "drug2": "Drug B",
  "cell_line": "Cell Line A",
  "gene_expression": [0.12, -0.43, 0.87, 0.21]
}
```

The actual feature requirements depend on the trained model and preprocessing pipeline.

---

## 📊 Model Evaluation

The trained models can be evaluated using appropriate machine-learning metrics, such as:

* Accuracy
* Precision
* Recall
* F1-score
* Mean Absolute Error
* Root Mean Squared Error
* R² score

The selected evaluation metrics depend on whether the final prediction task is treated as classification, regression, or both.

---

## 🌟 Future Enhancements

The project is designed to support several future improvements.

### 1. Explainable Predictions

Provide explanations for why a particular drug combination receives a specific prediction.

### 2. Drug Combination Ranking

Allow users to provide multiple candidate drug combinations and rank them according to predicted synergy.

### 3. What-If Gene Expression Analysis

Allow researchers to explore how changes in biological signatures affect the predicted synergy.

### 4. Confidence and Uncertainty Analysis

Provide additional information about model confidence or prediction uncertainty.

### 5. Molecular Feature Integration

Use molecular representations such as SMILES and molecular fingerprints to incorporate chemical structure information.

### 6. Interactive Research Dashboard

Extend the frontend into an interactive dashboard containing:

* Drug combination prediction
* Cell-line selection
* Gene expression analysis
* Prediction explanations
* Model performance
* Visualization of results

---

## ⚠️ Scientific Considerations

This system is intended for **computational research and educational purposes**.

A model prediction does not establish that a drug combination is clinically effective or safe. Computational predictions should be validated using appropriate experimental and scientific methods before drawing biological or clinical conclusions.

---

## 👥 Project Team:Chemix

**Project:** Drug-Synergy Prediction

**Focus:** Machine Learning, Gene Expression Analysis, and Oncology Drug Combination Prediction

**Repository:**
https://github.com/pyTharun/D-Synergy.git

---

## 📜 License

This project is intended for academic and educational purposes. A formal open-source license can be added as the project is finalized.
