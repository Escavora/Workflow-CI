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

# Setup tracking
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID')

if DAGSHUB_TOKEN:
    dagshub.init(repo_owner='Escavora',
                 repo_name='Eksperimen_SML_Alief_Athallah',
                 mlflow=True)
    print("Tracking: DagsHub")
else:
    print(f"Tracking: {mlflow.get_tracking_uri()}")

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'stroke_preprocessed.csv'))
print(f"Dataset loaded: {df.shape}")

# Split
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simpan dataset train dan test
X_train_df = X_train.copy(); X_train_df['stroke'] = y_train.values
X_test_df  = X_test.copy();  X_test_df['stroke']  = y_test.values
X_train_df.to_csv(os.path.join(BASE_DIR, 'dataset_train.csv'), index=False)
X_test_df.to_csv(os.path.join(BASE_DIR,  'dataset_test.csv'),  index=False)
print("dataset_train.csv dan dataset_test.csv disimpan.")

# Training
if not MLFLOW_RUN_ID:
    mlflow.set_experiment("stroke-prediction-ci")

with mlflow.start_run(run_id=MLFLOW_RUN_ID) as run:

    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi training set
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    y_test_prob  = model.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_test_pred, zero_division=0))
    mlflow.log_metric("recall",    recall_score(y_test, y_test_pred, zero_division=0))
    mlflow.log_metric("f1_score",  f1_score(y_test, y_test_pred, zero_division=0))
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_test_prob))

    # Confusion Matrix — Training
    cm_train = confusion_matrix(y_train, y_train_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_train, display_labels=['No Stroke', 'Stroke']).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Training")
    plt.tight_layout()
    cm_train_path = os.path.join(BASE_DIR, 'confusion_matrix_training.png')
    plt.savefig(cm_train_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_train_path)

    # Confusion Matrix — Testing
    cm_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_test, display_labels=['No Stroke', 'Stroke']).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Testing")
    plt.tight_layout()
    cm_test_path = os.path.join(BASE_DIR, 'confusion_matrix_testing.png')
    plt.savefig(cm_test_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_test_path)

    # Feature Importance — simpan sebagai CSV
    features = X.columns.tolist()
    fi_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    fi_path = os.path.join(BASE_DIR, 'feature_importance.csv')
    fi_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)
    print("feature_importance.csv disimpan.")

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Simpan run_id
    run_id = run.info.run_id
    with open(os.path.join(BASE_DIR, 'run_id.txt'), "w") as f:
        f.write(run_id)

    print(f"\nRun ID   : {run_id}")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_test_prob):.4f}")