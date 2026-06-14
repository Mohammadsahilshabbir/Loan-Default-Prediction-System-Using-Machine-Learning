import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

def generate_csv(path="credit_synthetic.csv", n_samples=10000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[
        "age", "income", "loan_amount", "loan_term",
        "num_of_credit_lines", "credit_history_length",
        "num_of_delinquencies", "credit_score", "savings", "employment_length"
    ])

    df["age"] = (df["age"] * 6 + 45).round().astype(int).clip(18, 90)
    df["income"] = (np.exp(df["income"]) * 2000).round(2)
    df["loan_amount"] = (np.abs(df["loan_amount"]) * 5000 + 1000).round(2)
    df["loan_term"] = (np.abs(df["loan_term"]) * 12 + 12).round().astype(int)
    df["num_of_credit_lines"] = (np.abs(df["num_of_credit_lines"]) * 3 + 1).round().astype(int)
    df["credit_history_length"] = (np.abs(df["credit_history_length"]) * 2 + 1).round().astype(int)
    df["num_of_delinquencies"] = (np.abs(df["num_of_delinquencies"]) * 1.5).round().astype(int)
    df["credit_score"] = (df["credit_score"] * 50 + 600).round().astype(int).clip(300, 850)
    df["savings"] = (np.abs(df["savings"]) * 10000).round(2)
    df["employment_length"] = (np.abs(df["employment_length"]) * 5).round().astype(int)

    df["default"] = y

    df.to_csv(path, index=False)
    print(f"Dataset generated: {path}")


if __name__ == "__main__":
    generate_csv()
