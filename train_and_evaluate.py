import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
import matplotlib.pyplot as plt
from joblib import dump
from generate_data import generate_csv

# Ensure output folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

CSV_PATH = "credit_synthetic.csv"

# Generate data if missing
if not os.path.exists(CSV_PATH):
    print("Generating synthetic dataset...")
    generate_csv(path=CSV_PATH)

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["default"])
y = df["default"]

X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
dump(rf, "models/random_forest.joblib")

# MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
dump(mlp, "models/mlp.joblib")

# Evaluate models
models = {"RandomForest": rf, "MLP": mlp}
results = []

plt.figure()
for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append((name, auc, acc))

    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.savefig("results/roc.png")
plt.close()

# Feature importances
importances = rf.feature_importances_
feat_names = X.columns.tolist()
order = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(importances)), importances[order])
plt.xticks(range(len(importances)), [feat_names[i] for i in order], rotation=45)
plt.tight_layout()
plt.title("Feature Importances (Random Forest)")
plt.savefig("results/feature_importances.png")
plt.close()

# Write metrics
with open("results/metrics.txt", "w") as f:
    for name, auc, acc in results:
        f.write(f"{name}: AUC={auc:.4f}, Accuracy={acc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, rf.predict(X_test)))

print("\nTraining complete!")
print("Check 'models/' and 'results/' folders.")

# # sample data
# print("Sample Prediction:")
# print(rf.predict([[45, 50000, 15000, 12, 3, 5, 0, 720, 2000, 2]]))
# #sample data
# print("Sample Prediction:")
# print(rf.predict([[45,50000,15000,13,4,6,1,721,2001,3]]))

# sample = pd.DataFrame([[
#     46, 50001, 15001, 13, 4, 6, 1, 721, 2001, 3
# ]], columns=X.columns)


sample_defaulter = pd.DataFrame([[
    22,      # age (very young)
    12000,   # income (very low)
    45000,   # loan_amount (very high)
    36,      # loan_duration (long)
    0,       # savings (none)
    1,       # credit_history (poor)
    1,       # previous_default (yes)
    420,     # credit_score (very low)
    9000,    # monthly_expenses (high)
    6        # dependents (many)
]], columns=X.columns)

# print("Sample Prediction:")
# print(rf.predict(sample_defaulter))
prob = rf.predict_proba(sample_defaulter)
print("No Default Probability:", prob[0][0])
print("Default Probability:", prob[0][1])
print("Sample Prediction:")
print(rf.predict(sample_defaulter))
