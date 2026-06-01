# Import library
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# Parse parameter dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',      type=int, default=100)
parser.add_argument('--max_depth',         type=int, default=5)
parser.add_argument('--min_samples_split', type=int, default=2)
args = parser.parse_args()

DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
if DAGSHUB_TOKEN:
    dagshub.init(repo_owner='Escavora',
                 repo_name='Eksperimen_SML_Alief_Athallah',
                 mlflow=True)
    print("Tracking: DagsHub")
else:
    # Pakai SQLite supaya kompatibel dengan MLflow versi baru
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("Tracking: SQLite lokal (mlflow.db)")

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'stroke_preprocessing', 'stroke_preprocessed.csv'))
print(f"Dataset loaded: {df.shape}")

# Split
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training + logging
mlflow.set_experiment("stroke-prediction-ci")

with mlflow.start_run(run_name="RF-MLProject") as run:

    # Log params
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_prob))

    # Artefak 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=['No Stroke', 'Stroke']).plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Artefak 2: Feature Importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    features = X.columns.tolist()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(features)), importances[idx], color='steelblue', edgecolor='white')
    plt.xticks(range(len(features)), [features[i] for i in idx], rotation=45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Simpan run_id untuk Docker build di CI
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"\nRun ID   : {run.info.run_id}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")