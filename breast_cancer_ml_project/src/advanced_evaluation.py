from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from config import (
    FEATURE_IMPORTANCE_FIGURE_FILE,
    RANDOM_STATE,
    SHAP_SUMMARY_FIGURE_FILE,
    THRESHOLD_FIGURE_FILE,
)
from src.evaluate import get_prediction_scores


MEDICAL_SCORING = {
    "accuracy": "accuracy",
    "precision_m": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall_m": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1_m": make_scorer(f1_score, pos_label=1, zero_division=0),
    "mcc": make_scorer(matthews_corrcoef),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def cross_validate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run 5-fold StratifiedKFold evaluation without changing fitted final models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for name, model in models.items():
        scores = cross_validate(
            clone(model),
            X,
            y,
            cv=cv,
            scoring=MEDICAL_SCORING,
            error_score=np.nan,
            n_jobs=-1,
        )
        rows.append(
            {
                "Model": name,
                "Accuracy Mean": float(np.nanmean(scores["test_accuracy"])),
                "Accuracy Std": float(np.nanstd(scores["test_accuracy"])),
                "Precision (M) Mean": float(np.nanmean(scores["test_precision_m"])),
                "Precision (M) Std": float(np.nanstd(scores["test_precision_m"])),
                "Recall (M) Mean": float(np.nanmean(scores["test_recall_m"])),
                "Recall (M) Std": float(np.nanstd(scores["test_recall_m"])),
                "F1-score (M) Mean": float(np.nanmean(scores["test_f1_m"])),
                "F1-score (M) Std": float(np.nanstd(scores["test_f1_m"])),
                "MCC Mean": float(np.nanmean(scores["test_mcc"])),
                "MCC Std": float(np.nanstd(scores["test_mcc"])),
                "ROC-AUC Mean": float(np.nanmean(scores["test_roc_auc"])),
                "ROC-AUC Std": float(np.nanstd(scores["test_roc_auc"])),
                "PR-AUC Mean": float(np.nanmean(scores["test_pr_auc"])),
                "PR-AUC Std": float(np.nanstd(scores["test_pr_auc"])),
            }
        )

    return pd.DataFrame(rows).set_index("Model")


def tune_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run small, reproducible GridSearchCV experiments for core models."""
    param_grids: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "max_depth": [3, 4, 5, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [2, 5],
        },
        "Random Forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
            "class_weight": [None, "balanced"],
        },
        "SVM": {
            "model__C": [0.5, 1.0, 2.0],
            "model__gamma": ["scale", "auto"],
            "model__class_weight": [None, "balanced"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for name, params in param_grids.items():
        if name not in models:
            continue
        search = GridSearchCV(
            estimator=clone(models[name]),
            param_grid=params,
            scoring="recall",
            cv=cv,
            n_jobs=-1,
            error_score=np.nan,
            refit=True,
        )
        search.fit(X_train, y_train)
        rows.append(
            {
                "Model": name,
                "Best CV Recall (M)": float(search.best_score_),
                "Best Params": search.best_params_,
                "Candidates": int(len(search.cv_results_["params"])),
                "Scoring": "recall",
            }
        )

    return pd.DataFrame(rows).set_index("Model")


def evaluate_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Evaluate probability thresholds from 0.30 to 0.60 for the positive class."""
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.30, 0.61, 0.05)]

    if not hasattr(model, "predict_proba"):
        return pd.DataFrame(), None

    y_score = model.predict_proba(X_test)[:, 1]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "Threshold": float(threshold),
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (M)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                "Recall (M)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                "F1-score (M)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                "Specificity (B)": specificity,
                "MCC": matthews_corrcoef(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_score),
                "PR-AUC": average_precision_score(y_test, y_score),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    threshold_df = pd.DataFrame(rows)
    ranked = threshold_df.sort_values(
        by=["Recall (M)", "FN", "F1-score (M)", "ROC-AUC", "Accuracy"],
        ascending=[False, True, False, False, False],
        kind="mergesort",
    )
    best_threshold = float(ranked.iloc[0]["Threshold"]) if not ranked.empty else None
    return threshold_df, best_threshold


def plot_threshold_optimization(threshold_df: pd.DataFrame, output_path: Path = THRESHOLD_FIGURE_FILE) -> Path | None:
    if threshold_df.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = threshold_df.melt(
        id_vars="Threshold",
        value_vars=["Precision (M)", "Recall (M)", "F1-score (M)", "Specificity (B)"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=plot_df, x="Threshold", y="Score", hue="Metric", marker="o")
    plt.ylim(0, 1.05)
    plt.title("Threshold Optimization for Malignant Recall")
    plt.xlabel("Decision threshold for Malignant probability")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _unwrap_pipeline(model, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        if len(model.steps) > 1:
            transformed = model[:-1].transform(X)
            transformed_df = pd.DataFrame(transformed, columns=X.columns, index=X.index)
            return estimator, transformed_df
        return estimator, X
    return model, X


def _positive_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] > 1:
        return values[:, :, 1]
    return values


def _feature_importance_values(estimator) -> np.ndarray | None:
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        return np.abs(coef)
    return None


def plot_feature_importance(
    estimator,
    feature_names: list[str],
    output_path: Path = FEATURE_IMPORTANCE_FIGURE_FILE,
    top_n: int = 15,
) -> tuple[Path | None, list[dict[str, Any]]]:
    importances = _feature_importance_values(estimator)
    if importances is None:
        return None, []

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    top_df = importance_df.head(top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_df, x="importance", y="feature")
    plt.title("Top Global Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path, top_df.to_dict(orient="records")


def generate_explainability_artifacts(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
) -> dict[str, Any]:
    """Generate SHAP summary when available plus a global feature-importance fallback."""
    estimator, X_train_model = _unwrap_pipeline(model, X_train)
    _, X_test_model = _unwrap_pipeline(model, X_test)
    figure_paths: list[str] = []
    errors: list[str] = []

    feature_importance_path, top_features = plot_feature_importance(estimator, feature_names)
    if feature_importance_path is not None:
        figure_paths.append(feature_importance_path.as_posix())

    shap_created = False
    try:
        import shap

        sample_size = min(120, len(X_test_model))
        X_sample = X_test_model.sample(sample_size, random_state=RANDOM_STATE)

        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = _positive_shap_values(explainer.shap_values(X_sample))
        elif hasattr(estimator, "coef_"):
            background = X_train_model.sample(min(120, len(X_train_model)), random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(estimator, background)
            shap_values = _positive_shap_values(explainer.shap_values(X_sample))
        else:
            y_score = get_prediction_scores(model, X_sample)
            if y_score is None:
                raise ValueError("Model does not expose probability or decision scores for SHAP.")
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = _positive_shap_values(explainer(X_sample).values)

        SHAP_SUMMARY_FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_FIGURE_FILE, dpi=180, bbox_inches="tight")
        plt.close()
        figure_paths.append(SHAP_SUMMARY_FIGURE_FILE.as_posix())
        shap_created = True
    except ModuleNotFoundError as exc:
        if exc.name == "shap":
            errors.append(
                "SHAP summary was not generated because the 'shap' package is not "
                "installed in the active Python environment. Run "
                "`python -m pip install -r requirements.txt`; the app keeps using "
                "global feature importance as a safe fallback."
            )
        else:
            errors.append(f"SHAP summary was not generated: {exc}")
    except Exception as exc:
        errors.append(
            f"SHAP summary was not generated: {exc}. "
            "Global feature importance is used as a safe fallback."
        )

    first_prediction_note = ""
    if top_features and not X_test.empty:
        first_row = X_test.iloc[0]
        highlights = []
        for item in top_features[:5]:
            feature = item["feature"]
            if feature in first_row:
                highlights.append(f"{feature}={first_row[feature]:.4f}")
        if highlights:
            first_prediction_note = (
                "For an example test record, influential global features include: "
                + ", ".join(highlights)
                + "."
            )

    return {
        "model": model_name,
        "shap_summary_created": shap_created,
        "top_features": top_features,
        "figure_paths": figure_paths,
        "errors": errors,
        "example_prediction_explanation": first_prediction_note,
        "medical_note": "Explanations support inspection only and do not replace clinical judgment.",
    }
