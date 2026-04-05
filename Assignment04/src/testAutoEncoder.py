import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

from pyod.models.auto_encoder import AutoEncoder

# Optional Kaggle download
USE_KAGGLEHUB = False

if USE_KAGGLEHUB:
    import kagglehub

RANDOM_STATE = 42
DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    if USE_KAGGLEHUB:
        print("[INFO] Downloading dataset with kagglehub...")
        path = kagglehub.dataset_download("whenamancodes/fraud-detection")
        csv_path = os.path.join(path, "creditcard.csv")
        print(f"[INFO] Dataset downloaded to: {csv_path}")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "Dataset not found. Put creditcard.csv in data/ or enable USE_KAGGLEHUB."
    )


def main():
    # 1) Load data
    df = load_dataset()
    print("\n[INFO] Dataset shape:", df.shape)
    print("[INFO] Class distribution:")
    print(df["Class"].value_counts())

    # 2) Features and labels
    X = df.drop(columns=["Class"])
    y = df["Class"].values

    # 3) Train/test split (stratified so fraud ratio stays similar)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4) Train AE only on NORMAL transactions
    X_train_normal = X_train_full[y_train_full == 0]

    # 5) Scale data
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    print("\n[INFO] Training samples (normal only):", X_train_normal_scaled.shape[0])
    print("[INFO] Test samples:", X_test_scaled.shape[0])
    print("[INFO] Frauds in test set:", int((y_test == 1).sum()))

    # 6) Build AutoEncoder
    # contamination = expected proportion of outliers in evaluation data
    clf = AutoEncoder(
        contamination=0.0017,   # about 0.17% fraud rate in dataset
        epoch_num=20,
        batch_size=256,
        hidden_neuron_list=[64, 32, 32, 64],
        dropout_rate=0.2,
        preprocessing=False,    # already scaled
        verbose=1,
        random_state=RANDOM_STATE
    )

    # 7) Train model
    clf.fit(X_train_normal_scaled)

    # 8) Outlier scores on test set
    y_scores = clf.decision_function(X_test_scaled)

    # PyOD threshold-based predictions
    y_pred = clf.predict(X_test_scaled)

    # 9) Metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    print("\n================ RESULTS ================")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 10) Save scores
    results_df = pd.DataFrame({
        "true_label": y_test,
        "anomaly_score": y_scores,
        "pred_label": y_pred
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "test_results.csv"), index=False)

    # 11) Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - AutoEncoder Fraud Detection")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 12) Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AutoEncoder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=200)
    plt.close()

    # 13) Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - AutoEncoder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=200)
    plt.close()

    print(f"\n[INFO] Saved files in '{OUTPUT_DIR}/':")
    print("- test_results.csv")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("=========================================\n")


if __name__ == "__main__":
    main()
