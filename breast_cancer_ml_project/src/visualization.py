from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.tree import plot_tree

from config import (
    BEST_MODEL_CONFUSION_MATRIX_FILE,
    CORRELATION_HEATMAP_FIGURE_FILE,
    DECISION_TREE_FIGURE_FILE,
    FIGURE_DIR,
    METRIC_COMPARISON_FIGURE_FILE,
    ROC_CURVE_FIGURE_FILE,
    TARGET_COL,
    TARGET_DISTRIBUTION_FIGURE_FILE,
)
from src.evaluate import get_prediction_scores


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("-", "_")


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> Path:
    return plot_confusion_matrix_to_file(
        model,
        X_test,
        y_test,
        model_name,
        FIGURE_DIR / f"confusion_matrix_{model_slug(model_name)}.png",
        f"Confusion Matrix - {model_name}",
    )


def plot_confusion_matrix_to_file(
    model,
    X_test,
    y_test,
    model_name: str,
    output_path: Path,
    title: str,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_best_model_confusion_matrix(model, X_test, y_test, model_name: str) -> Path:
    return plot_confusion_matrix_to_file(
        model,
        X_test,
        y_test,
        model_name,
        BEST_MODEL_CONFUSION_MATRIX_FILE,
        f"Confusion Matrix - Best Model: {model_name}",
    )


def plot_roc_curve(model, X_test, y_test, model_name: str) -> Path | None:
    y_score = get_prediction_scores(model, X_test)
    if y_score is None:
        return None

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    path = FIGURE_DIR / f"roc_curve_{model_slug(model_name)}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_combined_roc_curve(models: dict[str, object], X_test, y_test) -> Path | None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plotted = False
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_score = get_prediction_scores(model, X_test)
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})")
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_FIGURE_FILE, dpi=180)
    plt.close()
    return ROC_CURVE_FIGURE_FILE


def plot_metric_comparison(results_df: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision (M)", "Recall (M)", "F1-score (M)"]
    plot_df = results_df[metrics].reset_index()
    plot_df = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1.05)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Metric Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(METRIC_COMPARISON_FIGURE_FILE, dpi=180)
    plt.close()
    return METRIC_COMPARISON_FIGURE_FILE


def plot_target_distribution(df: pd.DataFrame, target_col: str = TARGET_COL) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values)
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of samples")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(TARGET_DISTRIBUTION_FIGURE_FILE, dpi=180)
    plt.close()
    return TARGET_DISTRIBUTION_FIGURE_FILE


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    corr_df = X.copy()
    corr_df["target"] = y.values
    correlations = corr_df.corr(numeric_only=True)["target"].drop("target").abs()
    top_features = correlations.sort_values(ascending=False).head(12).index.tolist()
    heatmap_df = corr_df[top_features + ["target"]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap - Top Features vs Target")
    plt.tight_layout()
    plt.savefig(CORRELATION_HEATMAP_FIGURE_FILE, dpi=180)
    plt.close()
    return CORRELATION_HEATMAP_FIGURE_FILE


def plot_decision_tree_model(model, feature_names: list[str]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Benign", "Malignant"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree for Breast Cancer Diagnosis Support")
    plt.tight_layout()
    plt.savefig(DECISION_TREE_FIGURE_FILE, dpi=180)
    plt.close()
    return DECISION_TREE_FIGURE_FILE
