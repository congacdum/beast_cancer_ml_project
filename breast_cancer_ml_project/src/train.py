from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BEST_MODEL_FILE,
    BEST_MODEL_SUMMARY_FILE,
    CV_RESULTS_FILE,
    CV_RESULTS_MD_FILE,
    DATA_DIR,
    DATA_FILE,
    DATA_QUALITY_FLAGS_FILE,
    DATA_QUALITY_HTML_FILE,
    DATA_QUALITY_REPORT_FILE,
    DECISION_THRESHOLD_FILE,
    DECISION_TREE_RULES_FILE,
    EXPERIMENT_SUMMARY_FILE,
    EXPLAINABILITY_REPORT_FILE,
    FEATURE_NAMES_FILE,
    FINAL_SUMMARY_FILE,
    FIGURE_DIR,
    LABEL_MAPPING_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_COMPARISON_MD_FILE,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    ROOT_DIR,
    TARGET_COL,
    TEST_SIZE,
    THRESHOLD_RESULTS_FILE,
    THRESHOLD_RESULTS_MD_FILE,
    TRAINING_METADATA_FILE,
    TUNING_RESULTS_FILE,
    TUNING_RESULTS_MD_FILE,
)
from src.advanced_evaluation import (
    cross_validate_models,
    evaluate_thresholds,
    generate_explainability_artifacts,
    plot_threshold_optimization,
    tune_models,
)
from src.data_utils import load_data as read_data
from src.data_validation import write_data_quality_flags, write_data_quality_html
from src.evaluate import evaluate_model, make_results_table
from src.preprocessing import preprocess_data as run_preprocessing
from src.visualization import (
    model_slug,
    plot_best_model_confusion_matrix,
    plot_combined_roc_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_decision_tree_model,
    plot_metric_comparison,
    plot_roc_curve,
    plot_target_distribution,
)


SIMPLE_MODEL_PRIORITY = {
    "Decision Tree": 0,
    "Logistic Regression": 1,
    "Naive Bayes": 2,
    "KNN": 3,
    "Random Forest": 4,
    "SVM": 5,
}

SELECTION_RULE = (
    "highest Recall (M), then lowest FN, then F1-score (M), then ROC-AUC, "
    "then Accuracy, then simpler/more interpretable model"
)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def load_data(data_path: Path = DATA_FILE) -> pd.DataFrame:
    return read_data(data_path)


def preprocess_data(df: pd.DataFrame, target_col: str | None = TARGET_COL) -> dict[str, Any]:
    return run_preprocessing(df, target_col=target_col)


def build_models() -> Dict[str, object]:
    """Create candidate models for Breast Cancer Classification."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_models(preprocessed: dict[str, Any]) -> Dict[str, object]:
    X_train = preprocessed["X_train"]
    y_train = preprocessed["y_train"]
    fitted_models = {}

    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models


def evaluate_models(
    models: Dict[str, object],
    preprocessed: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]], list[str], list[str]]:
    X_test = preprocessed["X_test"]
    y_test = preprocessed["y_test"]
    results = {}
    figure_paths: list[str] = []
    figure_errors: list[str] = []

    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
        try:
            figure_paths.append(display_path(plot_confusion_matrix(model, X_test, y_test, name)))
        except Exception as exc:
            figure_errors.append(f"confusion matrix {name}: {exc}")
        try:
            roc_path = plot_roc_curve(model, X_test, y_test, name)
            if roc_path is not None:
                figure_paths.append(display_path(roc_path))
        except Exception as exc:
            figure_errors.append(f"ROC curve {name}: {exc}")

    return make_results_table(results), results, figure_paths, figure_errors


def rank_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()
    ranked["_roc_auc_rank"] = ranked["ROC-AUC"].fillna(float("-inf"))
    ranked["_simplicity_rank"] = [
        SIMPLE_MODEL_PRIORITY.get(model_name, 999) for model_name in ranked.index
    ]
    ranked = ranked.sort_values(
        by=[
            "Recall (M)",
            "FN",
            "F1-score (M)",
            "_roc_auc_rank",
            "Accuracy",
            "_simplicity_rank",
        ],
        ascending=[False, True, False, False, False, True],
        kind="mergesort",
    )
    return ranked.drop(columns=["_roc_auc_rank", "_simplicity_rank"])


def select_best_model(
    results_df: pd.DataFrame,
    models: Dict[str, object],
) -> tuple[str, object, pd.DataFrame, str]:
    ranked = rank_results_table(results_df)
    best_name = ranked.index[0]
    best_row = ranked.loc[best_name]
    reason = (
        f"{best_name} was selected by the medical-priority rule: "
        f"Recall (M)={best_row['Recall (M)']:.4f}, FN={int(best_row['FN'])}, "
        f"F1-score (M)={best_row['F1-score (M)']:.4f}, "
        f"ROC-AUC={best_row['ROC-AUC']:.4f}, Accuracy={best_row['Accuracy']:.4f}. "
        "Recall for Malignant is prioritized because False Negative cases are more dangerous "
        "than False Positive cases in breast cancer screening."
    )
    return best_name, models[best_name], ranked, reason


def save_model(
    best_model,
    preprocessed: dict[str, Any],
    fitted_models: Dict[str, object],
    decision_threshold: float | None,
) -> dict[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, BEST_MODEL_FILE)
    joblib.dump(preprocessed["feature_names"], FEATURE_NAMES_FILE)
    joblib.dump(preprocessed["label_mapping"], LABEL_MAPPING_FILE)
    if decision_threshold is not None:
        joblib.dump(float(decision_threshold), DECISION_THRESHOLD_FILE)

    saved_files = {
        "best_model": display_path(BEST_MODEL_FILE),
        "feature_names": display_path(FEATURE_NAMES_FILE),
        "label_mapping": display_path(LABEL_MAPPING_FILE),
    }
    if decision_threshold is not None:
        saved_files["decision_threshold"] = display_path(DECISION_THRESHOLD_FILE)

    for name, model in fitted_models.items():
        model_path = MODEL_DIR / f"{model_slug(name)}.joblib"
        joblib.dump(model, model_path)
        saved_files[f"model_{model_slug(name)}"] = display_path(model_path)

    return saved_files


def format_metric_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    work = df.copy()
    if index:
        index_name = work.index.name or "Index"
        work.insert(0, index_name, work.index)
    columns = list(work.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(work[column]) else "---" for column in columns) + "|"
    rows = [header, separator]
    for _, row in work.iterrows():
        values = []
        for column in columns:
            if str(column) in {"TN", "FP", "FN", "TP", "Candidates"} and pd.notna(row[column]):
                values.append(str(int(row[column])))
            else:
                values.append(format_metric_value(row[column]).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def results_to_markdown(results_df: pd.DataFrame) -> str:
    columns = [
        "Accuracy",
        "Precision (M)",
        "Recall (M)",
        "F1-score (M)",
        "Specificity (B)",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    header = "| Model | " + " | ".join(columns) + " |"
    separator = "|---|" + "|".join(["---:" for _ in columns]) + "|"
    rows = [header, separator]

    for model_name, row in results_df.iterrows():
        values = []
        for column in columns:
            if column in {"TN", "FP", "FN", "TP"}:
                values.append(str(int(row[column])))
            else:
                values.append(format_metric_value(row[column]))
        rows.append(f"| {model_name} | " + " | ".join(values) + " |")

    return "\n".join(rows)


def save_markdown_table(df: pd.DataFrame, output_path: Path, index: bool = True) -> str:
    table = dataframe_to_markdown(df, index=index)
    output_path.write_text(table + "\n", encoding="utf-8")
    return table


def save_model_comparison_markdown(results_df: pd.DataFrame) -> str:
    table = results_to_markdown(results_df)
    MODEL_COMPARISON_MD_FILE.write_text(table + "\n", encoding="utf-8")
    return table


def add_figure(
    figure_paths: list[str],
    figure_errors: list[str],
    label: str,
    action: Callable[[], Path | None],
) -> None:
    try:
        path = action()
        if path is not None:
            figure_paths.append(display_path(path))
    except Exception as exc:
        figure_errors.append(f"{label}: {exc}")


def create_report_figures(
    df: pd.DataFrame,
    preprocessed: dict[str, Any],
    fitted_models: Dict[str, object],
    best_name: str,
    best_model,
    results_df: pd.DataFrame,
    figure_errors: list[str],
) -> list[str]:
    figure_paths: list[str] = []

    add_figure(
        figure_paths,
        figure_errors,
        "target distribution",
        lambda: plot_target_distribution(df, preprocessed["target_col"]),
    )
    add_figure(
        figure_paths,
        figure_errors,
        "correlation heatmap",
        lambda: plot_correlation_heatmap(preprocessed["X"], preprocessed["y"]),
    )
    add_figure(
        figure_paths,
        figure_errors,
        "best model confusion matrix",
        lambda: plot_best_model_confusion_matrix(
            best_model,
            preprocessed["X_test"],
            preprocessed["y_test"],
            best_name,
        ),
    )
    add_figure(
        figure_paths,
        figure_errors,
        "model metric comparison",
        lambda: plot_metric_comparison(results_df),
    )
    add_figure(
        figure_paths,
        figure_errors,
        "combined ROC curve",
        lambda: plot_combined_roc_curve(
            fitted_models,
            preprocessed["X_test"],
            preprocessed["y_test"],
        ),
    )
    return figure_paths


def export_decision_tree_artifacts(
    fitted_models: Dict[str, object],
    feature_names: list[str],
    figure_paths: list[str],
    figure_errors: list[str],
) -> None:
    if "Decision Tree" not in fitted_models:
        return

    dt_model = fitted_models["Decision Tree"]
    rules = export_text(dt_model, feature_names=feature_names)
    DECISION_TREE_RULES_FILE.write_text(rules, encoding="utf-8")
    add_figure(
        figure_paths,
        figure_errors,
        "decision tree",
        lambda: plot_decision_tree_model(dt_model, feature_names),
    )


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def best_row_to_dict(best_row: pd.Series) -> dict[str, float | int | None]:
    result = {}
    for column in best_row.index:
        if column in {"TN", "FP", "FN", "TP"}:
            result[column] = int(best_row[column])
        else:
            result[column] = json_ready(best_row[column])
    return result


def build_explainability_markdown(explainability: dict[str, Any]) -> str:
    top_features = explainability.get("top_features", [])
    top_lines = "\n".join(
        f"- {item['feature']}: {float(item['importance']):.6f}" for item in top_features[:15]
    ) or "- No feature importance values available."
    figure_lines = "\n".join(f"- `{display_path(Path(path))}`" for path in explainability.get("figure_paths", [])) or "- None"
    error_lines = "\n".join(f"- {error}" for error in explainability.get("errors", [])) or "- None"

    return f"""# Model Explainability

## Model

- Model: {explainability.get("model", "N/A")}
- SHAP summary created: {explainability.get("shap_summary_created", False)}

## Top Global Features

{top_lines}

## Example Prediction Explanation

{explainability.get("example_prediction_explanation") or "No example explanation was generated."}

## Figures

{figure_lines}

## Notes / Errors

{error_lines}

Medical note: explanations support inspection only and do not replace clinical judgment.
"""


def save_reports(
    preprocessed: dict[str, Any],
    results_df: pd.DataFrame,
    markdown_table: str,
    cv_df: pd.DataFrame,
    cv_markdown: str,
    tuning_df: pd.DataFrame,
    tuning_markdown: str,
    threshold_df: pd.DataFrame,
    threshold_markdown: str,
    decision_threshold: float | None,
    best_name: str,
    selection_reason: str,
    saved_files: dict[str, str],
    figure_paths: list[str],
    figure_errors: list[str],
    explainability: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_report = preprocessed["data_report"]
    DATA_QUALITY_REPORT_FILE.write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_data_quality_html(data_report, DATA_QUALITY_HTML_FILE)
    write_data_quality_flags(
        preprocessed["cleaned_df"],
        preprocessed["outlier_flags"],
        DATA_QUALITY_FLAGS_FILE,
    )

    best_row = results_df.loc[best_name]
    EXPLAINABILITY_REPORT_FILE.write_text(
        build_explainability_markdown(explainability),
        encoding="utf-8",
    )

    metadata = {
        "dataset": display_path(DATA_FILE),
        "target_column": preprocessed["target_col"],
        "target_mapping": {"Benign": 0, "Malignant": 1},
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
        "best_model": best_name,
        "selection_rule": SELECTION_RULE,
        "selection_reason": selection_reason,
        "cross_validation_note": (
            "Cross-validation is run on the training split only for stability review. "
            "The final saved model is selected from the untouched hold-out table by "
            "the fixed medical-first rule."
        ),
        "best_model_metrics": best_row_to_dict(best_row),
        "decision_threshold": decision_threshold,
        "decision_threshold_source": "training_split",
        "saved_files": saved_files,
        "reports": {
            "model_comparison_csv": display_path(MODEL_COMPARISON_FILE),
            "model_comparison_md": display_path(MODEL_COMPARISON_MD_FILE),
            "cross_validation_csv": display_path(CV_RESULTS_FILE),
            "cross_validation_md": display_path(CV_RESULTS_MD_FILE),
            "tuning_csv": display_path(TUNING_RESULTS_FILE),
            "tuning_md": display_path(TUNING_RESULTS_MD_FILE),
            "threshold_csv": display_path(THRESHOLD_RESULTS_FILE),
            "threshold_md": display_path(THRESHOLD_RESULTS_MD_FILE),
            "data_quality_json": display_path(DATA_QUALITY_REPORT_FILE),
            "data_quality_html": display_path(DATA_QUALITY_HTML_FILE),
            "explainability_md": display_path(EXPLAINABILITY_REPORT_FILE),
        },
        "figures": figure_paths,
        "figure_errors": figure_errors,
        "explainability": explainability,
        "label_mapping": preprocessed["label_mapping"],
        "note": "The model is for educational diagnosis support only, not a medical device.",
    }
    TRAINING_METADATA_FILE.write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    BEST_MODEL_SUMMARY_FILE.write_text(
        build_best_model_summary(
            best_name,
            selection_reason,
            best_row,
            threshold_df,
            decision_threshold,
        ),
        encoding="utf-8",
    )
    experiment_summary = build_experiment_summary(
        preprocessed,
        markdown_table,
        cv_markdown,
        tuning_markdown,
        threshold_markdown,
        best_name,
        best_row,
        decision_threshold,
        figure_paths,
        saved_files,
        figure_errors,
        explainability,
    )
    EXPERIMENT_SUMMARY_FILE.write_text(experiment_summary, encoding="utf-8")
    FINAL_SUMMARY_FILE.write_text(experiment_summary, encoding="utf-8")


def build_best_model_summary(
    best_name: str,
    selection_reason: str,
    best_row: pd.Series,
    threshold_df: pd.DataFrame,
    decision_threshold: float | None,
) -> str:
    threshold_note = "Threshold optimization was not available for this model."
    if decision_threshold is not None and not threshold_df.empty:
        row = threshold_df.loc[threshold_df["Threshold"].sub(decision_threshold).abs().idxmin()]
        threshold_note = (
            f"Optimized decision threshold on the training split: {decision_threshold:.2f}. "
            f"At this threshold, Recall (M)={row['Recall (M)']:.4f}, "
            f"FN={int(row['FN'])}, FP={int(row['FP'])}."
        )

    return f"""# Best Model Summary

## Best Model

- Best model: {best_name}
- Selection rule: {SELECTION_RULE}
- Reason: {selection_reason}

## Metrics at Default Model Decision Rule

| Metric | Value |
|---|---:|
| Accuracy | {format_metric_value(best_row["Accuracy"])} |
| Precision (M) | {format_metric_value(best_row["Precision (M)"])} |
| Recall (M) | {format_metric_value(best_row["Recall (M)"])} |
| F1-score (M) | {format_metric_value(best_row["F1-score (M)"])} |
| Specificity (B) | {format_metric_value(best_row["Specificity (B)"])} |
| MCC | {format_metric_value(best_row["MCC"])} |
| ROC-AUC | {format_metric_value(best_row["ROC-AUC"])} |
| PR-AUC | {format_metric_value(best_row["PR-AUC"])} |

## Confusion Matrix

- TN: {int(best_row["TN"])}
- FP: {int(best_row["FP"])}
- FN: {int(best_row["FN"])}
- TP: {int(best_row["TP"])}

## Threshold Optimization

{threshold_note}

## False Negative Note

FN is the number of malignant cases predicted as benign. In medical screening, FN is more
dangerous than FP because it can delay follow-up examination or treatment. This is why
model selection prioritizes Recall (M) and FN before general Accuracy.

## Medical Warning

This model is for learning/demo purposes only. It does not replace a doctor, clinical
testing, or formal medical diagnosis.
"""


def build_experiment_summary(
    preprocessed: dict[str, Any],
    markdown_table: str,
    cv_markdown: str,
    tuning_markdown: str,
    threshold_markdown: str,
    best_name: str,
    best_row: pd.Series,
    decision_threshold: float | None,
    figure_paths: list[str],
    saved_files: dict[str, str],
    figure_errors: list[str],
    explainability: dict[str, Any],
) -> str:
    data_report = preprocessed["data_report"]
    model_list = "\n".join(f"- {name}" for name in build_models().keys())
    figure_list = "\n".join(f"- `{path}`" for path in figure_paths) or "- None"
    artifact_list = "\n".join(f"- `{path}`" for path in saved_files.values())
    figure_error_text = "\n".join(f"- {error}" for error in figure_errors) or "- None"
    top_corr = data_report["correlation"]["highly_correlated_pairs"][:10]
    corr_text = "\n".join(
        f"- {item['feature_1']} vs {item['feature_2']}: {item['abs_correlation']:.4f}"
        for item in top_corr
    ) or "- No feature pairs above |corr| > 0.95."
    missing_actions = "\n".join(
        f"- {item.get('column', 'rows')}: {item.get('strategy')} ({item.get('reason')})"
        for item in data_report["imputation_decisions"]
    ) or "- No missing values required imputation."
    threshold_text = (
        f"{decision_threshold:.2f}" if decision_threshold is not None else "N/A"
    )
    shap_status = "created" if explainability.get("shap_summary_created") else "not created"
    return f"""# Experiment Summary

## Dataset

- Dataset: `{display_path(DATA_FILE)}`
- Samples: {data_report["shape"]["rows"]}
- Features before encoding: {data_report["shape"]["columns"] - 1}
- Features after encoding: {data_report["feature_count_after_encoding"]}
- Target column: `{preprocessed["target_col"]}`
- Target mapping:
  - Benign = 0
  - Malignant = 1

## Data Quality

- Quality score: {data_report["quality_score"]}/100
- Duplicate rows: {data_report["duplicate_rows"]}
- Suspected outliers: {data_report["outliers"]["total_suspected_outliers"]}
- Class balance: {data_report["class_balance"]["counts"]}
- Class imbalance detected: {data_report["class_balance"]["imbalance_detected"]}

### Missing / Cleaning Decisions

{missing_actions}

### Highly Correlated Features

{corr_text}

Outliers and highly correlated features are reported for review and are not deleted automatically.

## Train/Test Split

- Test size: {TEST_SIZE}
- Random state: {RANDOM_STATE}
- Stratify: `y`
- Scaling: model pipelines fit `StandardScaler` on training folds/train set only.

## Models

{model_list}

## Hold-out Metric Table

{markdown_table}

## 5-Fold Stratified Cross-Validation

{cv_markdown}

## Hyperparameter Tuning

GridSearchCV was run for Logistic Regression, Decision Tree, Random Forest, SVM, and KNN
using Recall (M) as the tuning score. These tuning results are saved as experiment evidence
and do not silently replace the fixed baseline models.

{tuning_markdown}

## Best Model

- Best model: {best_name}
- Selection rule: {SELECTION_RULE}
- Decision threshold used by prediction artifacts: {threshold_text}
- Note: cross-validation is run on the training split only. The final deployment model is
  selected from the untouched hold-out evaluation table and does not replace the fixed final
  hold-out selection rule.

## Confusion Matrix of Best Model

- TN: {int(best_row["TN"])}
- FP: {int(best_row["FP"])}
- FN: {int(best_row["FN"])}
- TP: {int(best_row["TP"])}

## Threshold Optimization

Threshold optimization is computed on the training split only. The hold-out test split remains
reserved for final model evaluation and model selection reporting.

{threshold_markdown}

## Explainability

- SHAP summary: {shap_status}
- Explainability report: `{display_path(EXPLAINABILITY_REPORT_FILE)}`

## Figures

{figure_list}

## Model / Artifact Files

{artifact_list}

## Figure Generation Errors

{figure_error_text}

## Limitations

- No nested cross-validation yet.
- Hyperparameter tuning grid is intentionally small for classroom/demo runtime.
- Pytest covers core preprocessing, validation, and prediction paths; UI tests are still manual.
- SHAP can be skipped if the optional package/runtime cannot support it.

## Future Work

- Add 5-fold nested cross-validation for tuned models.
- Add broader GridSearchCV/RandomizedSearchCV.
- Expand pytest coverage for training report generation and Streamlit utility flows.
- Add SHAP/LIME explanations in the Streamlit prediction page.

## Medical Warning

This model is for learning/demo purposes only. Results do not replace medical advice,
clinical tests, or a doctor's diagnosis.
"""


def run_pipeline() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    preprocessed = preprocess_data(df)
    fitted_models = train_models(preprocessed)
    results_df, results, figure_paths, figure_errors = evaluate_models(fitted_models, preprocessed)
    best_name, best_model, ranked_results_df, selection_reason = select_best_model(
        results_df,
        fitted_models,
    )

    cv_df = cross_validate_models(build_models(), preprocessed["X_train"], preprocessed["y_train"])
    tuning_df = tune_models(build_models(), preprocessed["X_train"], preprocessed["y_train"])
    threshold_df, decision_threshold = evaluate_thresholds(
        best_model,
        preprocessed["X_train"],
        preprocessed["y_train"],
    )

    ranked_results_df.to_csv(MODEL_COMPARISON_FILE, index=True)
    markdown_table = save_model_comparison_markdown(ranked_results_df)
    cv_df.to_csv(CV_RESULTS_FILE, index=True)
    cv_markdown = save_markdown_table(cv_df, CV_RESULTS_MD_FILE, index=True)
    tuning_df.to_csv(TUNING_RESULTS_FILE, index=True)
    tuning_markdown = save_markdown_table(tuning_df, TUNING_RESULTS_MD_FILE, index=True)
    threshold_df.to_csv(THRESHOLD_RESULTS_FILE, index=False)
    threshold_markdown = save_markdown_table(threshold_df, THRESHOLD_RESULTS_MD_FILE, index=False)

    if decision_threshold is not None:
        add_figure(
            figure_paths,
            figure_errors,
            "threshold optimization",
            lambda: plot_threshold_optimization(threshold_df),
        )

    explainability = generate_explainability_artifacts(
        best_model,
        preprocessed["X_train"],
        preprocessed["X_test"],
        preprocessed["feature_names"],
        best_name,
    )
    for path_text in explainability.get("figure_paths", []):
        figure_paths.append(display_path(Path(path_text)))
    figure_errors.extend(explainability.get("errors", []))

    saved_files = save_model(
        best_model,
        preprocessed,
        fitted_models,
        decision_threshold,
    )

    figure_paths.extend(create_report_figures(
        df,
        preprocessed,
        fitted_models,
        best_name,
        best_model,
        ranked_results_df,
        figure_errors,
    ))
    export_decision_tree_artifacts(
        fitted_models,
        preprocessed["feature_names"],
        figure_paths,
        figure_errors,
    )
    save_reports(
        preprocessed,
        ranked_results_df,
        markdown_table,
        cv_df,
        cv_markdown,
        tuning_df,
        tuning_markdown,
        threshold_df,
        threshold_markdown,
        decision_threshold,
        best_name,
        selection_reason,
        saved_files,
        figure_paths,
        figure_errors,
        explainability,
    )

    return {
        "models": fitted_models,
        "results": results,
        "results_df": ranked_results_df,
        "cv_results_df": cv_df,
        "tuning_results_df": tuning_df,
        "threshold_results_df": threshold_df,
        "decision_threshold": decision_threshold,
        "best_model_name": best_name,
        "best_model": best_model,
        "selection_reason": selection_reason,
        "saved_files": saved_files,
        "figure_errors": figure_errors,
    }


def main() -> None:
    pipeline_result = run_pipeline()
    results_df = pipeline_result["results_df"]
    best_name = pipeline_result["best_model_name"]

    print("\nTraining completed.")
    print(f"Dataset loaded from: {display_path(DATA_FILE)}")
    print(f"Data report saved to: {display_path(DATA_QUALITY_REPORT_FILE)}")
    print(f"Data quality HTML saved to: {display_path(DATA_QUALITY_HTML_FILE)}")
    print(f"Results saved to: {display_path(MODEL_COMPARISON_FILE)}")
    print(f"Markdown results saved to: {display_path(MODEL_COMPARISON_MD_FILE)}")
    print(f"Cross-validation saved to: {display_path(CV_RESULTS_FILE)}")
    print(f"Tuning results saved to: {display_path(TUNING_RESULTS_FILE)}")
    print(f"Threshold optimization saved to: {display_path(THRESHOLD_RESULTS_FILE)}")
    print(f"Best model: {best_name}")
    print(f"Decision threshold: {pipeline_result['decision_threshold']}")
    print(f"Selection rule: {SELECTION_RULE}")
    print(f"Selection reason: {pipeline_result['selection_reason']}")
    print(results_df.round(4))

    if pipeline_result["figure_errors"]:
        print("\nFigure/explainability warnings:")
        for error in pipeline_result["figure_errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
