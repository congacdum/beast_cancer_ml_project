from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_prediction_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def evaluate_model(model, X_test, y_test) -> Dict[str, float | int]:
    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0
    metrics: Dict[str, float | int] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (M)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (M)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-score (M)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Specificity (B)": specificity,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ROC-AUC": float("nan"),
        "PR-AUC": float("nan"),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if y_score is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_score)
        metrics["PR-AUC"] = average_precision_score(y_test, y_score)

    return metrics


def make_results_table(results: Dict[str, Dict[str, float | int]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    for column in ["TN", "FP", "FN", "TP"]:
        df[column] = df[column].astype(int)
    return df
