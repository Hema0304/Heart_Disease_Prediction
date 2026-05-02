import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42

# Updated color palette (blue-friendly with variety)
PALETTE = [
    "#1f77b4",  # Blue
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#17becf",  # Cyan
    "#d62728"   # Red
]

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y

def load_selected_features(path="results/selected_features.txt"):
    subsets = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(":")[1]
            clean = parts.replace("np.int64(", "").replace(")", "").replace("[", "").replace("]", "")
            indices = list(map(int, clean.strip().split(",")))
            subsets.append(indices)
    return subsets

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "predictions": preds,
        "probabilities": probs
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, subset_idx):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')  # Changed to green theme
    plt.title(f"Confusion Matrix: {model_name} (Subset {subset_idx+1})")
    os.makedirs("results/confusion_matrices", exist_ok=True)
    plt.savefig(f"results/confusion_matrices/subset_{subset_idx+1}_{model_name}.png")
    plt.close()

def plot_combined_roc(roc_data):
    os.makedirs("results/graphs", exist_ok=True)
    plt.figure(figsize=(10, 8))
    for i, (model_name, fpr, tpr) in enumerate(roc_data):
        plt.plot(fpr, tpr, label=model_name, color=PALETTE[i % len(PALETTE)], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curve")
    plt.legend()
    plt.savefig("results/graphs/roc_curves.png")
    plt.close()

def plot_performance_comparison(perf_results):
    os.makedirs("results/graphs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for m_idx, metric in enumerate(["accuracy", "f1_score", "roc_auc"]):
        for i, model in enumerate(perf_results[0]):
            values = [res[model][metric] for res in perf_results]
            ax.plot(
                range(1, len(values)+1), 
                values, 
                marker='o', 
                label=f"{model} - {metric}",
                color=PALETTE[(m_idx + i) % len(PALETTE)]
            )
    ax.set_title("Model Performance Across ACO Feature Subsets")
    ax.set_xlabel("Subset Index")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(ticks=range(1, len(perf_results)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results/graphs/performance.png")
    plt.close()
