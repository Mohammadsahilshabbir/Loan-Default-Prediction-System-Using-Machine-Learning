📌 Project Overview

This project predicts whether a customer is likely to default on a loan using Machine Learning techniques. Two models were trained and compared:

Random Forest Classifier
Multi-Layer Perceptron (MLP) Neural Network

The project includes synthetic credit data generation, model training, evaluation, feature importance analysis, and ROC curve visualization. The objective is to help financial institutions identify high-risk borrowers and support better lending decisions.

🚀 Features
Synthetic credit dataset generation
Data preprocessing and cleaning
Random Forest and MLP model training
Performance evaluation using Accuracy and AUC-ROC
Feature Importance Analysis
ROC Curve Comparison
Model serialization using Joblib
🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Joblib
📂 Dataset Features

The dataset contains the following borrower attributes:

Feature	Description
Age	Applicant's age
Income	Annual income
Loan Amount	Requested loan amount
Loan Term	Loan duration
Number of Credit Lines	Total active credit lines
Credit History Length	Length of credit history
Number of Delinquencies	Previous delayed payments
Credit Score	Creditworthiness score
Savings	Available savings
Employment Length	Years of employment

Target Variable:

Default

0 → No Default
1 → Default
📊 Model Performance
Model	AUC Score	Accuracy
Random Forest	0.9500	89.90%
MLP Neural Network	0.6242	70.60%

The Random Forest model significantly outperformed the MLP model and was selected as the best-performing model.

📈 Results
ROC Curve Analysis

The ROC curve compares both models:

Random Forest achieved an excellent AUC of 0.95
MLP achieved an AUC of 0.62
Random Forest demonstrates much stronger classification capability
Feature Importance

The most influential factors affecting loan default prediction were:

Income
Age
Savings
Loan Amount
Loan Term

These features contributed the most toward the model's decision-making process.

📋 Classification Report (Random Forest)
Metric	Class 0	Class 1
Precision	0.90	0.89
Recall	0.96	0.76
F1-Score	0.93	0.82

Overall Accuracy: 89.9%

## ⚙️ Installation

```bash
git clone https://github.com/Mohammadsahilshabbir/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Generate Dataset

```bash
python generate_data.py
```

### Train Models

```bash
python train_and_evaluate.py
```

---

## 📁 Project Structure

```text
Loan-Default-Prediction/
│
├── credit_synthetic.csv
├── generate_data.py
├── train_and_evaluate.py
├── requirements.txt
│
├── models/
│   ├── random_forest.joblib
│   └── mlp.joblib
│
└── results/
    ├── roc.png
    ├── feature_importances.png
    └── metrics.txt
```

---

## 🚀 Future Improvements

- Train on real-world financial datasets
- Hyperparameter tuning using Grid Search
- Deploy as a Flask/Django web application
- Integrate Explainable AI (SHAP/LIME)
- Experiment with XGBoost and LightGBM

---

## 👨‍💻 Author

**Mohammad Sahil Shabbir**

Machine Learning project focused on loan risk assessment and credit default prediction using supervised learning algorithms.
