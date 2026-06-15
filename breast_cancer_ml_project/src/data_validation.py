from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


MISSING_TOKENS = {
    "",
    " ",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
    "?",
}

BENIGN_TOKENS = {"b", "benign", "lành tính", "lanh tinh", "0"}
MALIGNANT_TOKENS = {"m", "malignant", "ác tính", "ac tinh", "1"}
NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _safe_ratio(count: int | float, total: int | float) -> float:
    return float(count / total) if total else 0.0


def normalize_missing_tokens(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert common missing-value strings to pandas NA."""
    cleaned = df.copy()
    replacements: dict[str, int] = {}

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[column]
        mask = series.map(
            lambda value: isinstance(value, str)
            and value.strip().casefold() in MISSING_TOKENS
        )
        count = int(mask.sum())
        if count:
            cleaned.loc[mask, column] = pd.NA
            replacements[column] = count

    return cleaned, {"missing_token_replacements": replacements}


def normalize_text_values(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trim text values and normalize common label variants."""
    cleaned = df.copy()
    normalized_columns: dict[str, int] = {}
    label_replacements: dict[str, dict[str, int]] = {}

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        original = cleaned[column]
        trimmed = original.map(lambda value: value.strip() if isinstance(value, str) else value)
        changed = int((trimmed.astype("string") != original.astype("string")).fillna(False).sum())
        if changed:
            normalized_columns[column] = changed
        cleaned[column] = trimmed

        if target_col and column == target_col:
            counts = {"benign": 0, "malignant": 0}

            def normalize_label(value):
                if not isinstance(value, str):
                    return value
                token = value.strip().casefold()
                if token in BENIGN_TOKENS:
                    counts["benign"] += 1
                    return "benign"
                if token in MALIGNANT_TOKENS:
                    counts["malignant"] += 1
                    return "malignant"
                return value

            cleaned[column] = cleaned[column].map(normalize_label)
            if any(counts.values()):
                label_replacements[column] = counts

    return cleaned, {
        "trimmed_text_values": normalized_columns,
        "label_replacements": label_replacements,
    }


def normalize_numeric_strings(
    df: pd.DataFrame,
    skip_columns: set[str] | None = None,
    min_parse_ratio: float = 0.85,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert numeric-looking strings such as '12.3 mm' to numeric values."""
    cleaned = df.copy()
    skip_columns = skip_columns or set()
    conversions: dict[str, Any] = {}

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        if column in skip_columns:
            continue

        series = cleaned[column]
        non_missing = series.dropna()
        if non_missing.empty:
            continue

        extracted = series.astype("string").str.extract(f"({NUMERIC_PATTERN.pattern})", expand=False)
        numeric = pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")
        parse_ratio = _safe_ratio(int(numeric.notna().sum()), int(non_missing.shape[0]))

        if parse_ratio >= min_parse_ratio:
            cleaned[column] = numeric
            conversions[column] = {
                "parse_ratio": parse_ratio,
                "non_null_values": int(non_missing.shape[0]),
                "parsed_values": int(numeric.notna().sum()),
                "note": "Converted numeric strings and stripped simple text units.",
            }

    return cleaned, {"numeric_string_conversions": conversions}


def normalize_dataframe(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply lightweight, reportable cleaning before preprocessing."""
    normalized, missing_report = normalize_missing_tokens(df)
    normalized, text_report = normalize_text_values(normalized, target_col=target_col)
    normalized, numeric_report = normalize_numeric_strings(
        normalized,
        skip_columns={target_col} if target_col else set(),
    )
    return normalized, {
        **missing_report,
        **text_report,
        **numeric_report,
    }


def validate_schema(
    df: pd.DataFrame,
    target_col: str,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    report = {
        "columns": list(df.columns),
        "column_count": int(df.shape[1]),
        "target_column": target_col,
        "target_present": target_col in df.columns,
        "missing_expected_columns": [],
        "extra_columns": [],
    }

    if expected_columns:
        actual_columns = set(df.columns)
        expected = set(expected_columns)
        report["expected_columns"] = expected_columns
        report["missing_expected_columns"] = sorted(expected - actual_columns)
        report["extra_columns"] = sorted(actual_columns - expected)

    return report


def validate_data_types(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    numeric_columns = [
        column for column in df.select_dtypes(include="number").columns if column != target_col
    ]
    text_columns = [
        column for column in df.select_dtypes(include=["object", "string"]).columns
        if column != target_col
    ]
    infinite_counts = {}
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        infinite_counts[column] = int(np.isinf(values.to_numpy(dtype=float)).sum())

    return {
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_feature_columns": numeric_columns,
        "text_feature_columns": text_columns,
        "infinite_value_counts": infinite_counts,
    }


def summarize_missing_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    total = int(df.shape[0])
    rows = []
    for column, count in df.isna().sum().items():
        rows.append(
            {
                "column": column,
                "missing_count": int(count),
                "missing_percent": _safe_ratio(int(count), total),
            }
        )
    return rows


def summarize_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_mask = df.duplicated(keep=False)
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_including_first_occurrence": int(duplicate_mask.sum()),
        "sample_duplicate_indices": [int(index) for index in df.index[duplicate_mask].tolist()[:20]],
    }


def validate_ranges(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Flag invalid numeric ranges; do not modify data."""
    numeric_columns = [
        column for column in df.select_dtypes(include="number").columns if column != target_col
    ]
    invalid_ranges: dict[str, Any] = {}

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        negative_mask = series < 0
        invalid = int(negative_mask.sum())
        if invalid:
            invalid_ranges[column] = {
                "rule": "Expected non-negative numeric value for breast cancer measurements.",
                "invalid_count": invalid,
                "sample_indices": [int(index) for index in series.index[negative_mask].tolist()[:20]],
            }

        if "age" in column.casefold():
            age_mask = series.notna() & ~series.between(0, 120)
            age_invalid = int(age_mask.sum())
            if age_invalid:
                invalid_ranges[column] = {
                    "rule": "Expected age between 0 and 120.",
                    "invalid_count": age_invalid,
                    "sample_indices": [int(index) for index in series.index[age_mask].tolist()[:20]],
                }

    return invalid_ranges


def detect_outliers(
    X: pd.DataFrame,
    random_state: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Flag suspected outliers with IQR, Z-score, and IsolationForest."""
    numeric_X = X.select_dtypes(include="number").copy()
    if numeric_X.empty:
        empty_flags = pd.Series(False, index=X.index, name="suspected_outlier")
        return empty_flags, {"methods": {}, "total_suspected_outliers": 0}

    numeric_X = numeric_X.replace([np.inf, -np.inf], np.nan)
    numeric_X = numeric_X.fillna(numeric_X.median(numeric_only=True))
    numeric_X = numeric_X.fillna(0)

    q1 = numeric_X.quantile(0.25)
    q3 = numeric_X.quantile(0.75)
    iqr = q3 - q1
    iqr_mask = ((numeric_X < (q1 - 1.5 * iqr)) | (numeric_X > (q3 + 1.5 * iqr))).any(axis=1)
    iqr_feature_counts = (
        ((numeric_X < (q1 - 1.5 * iqr)) | (numeric_X > (q3 + 1.5 * iqr))).sum().astype(int).to_dict()
    )

    std = numeric_X.std(ddof=0).replace(0, np.nan)
    z_scores = ((numeric_X - numeric_X.mean()) / std).abs()
    z_mask = z_scores.gt(3).any(axis=1).fillna(False)
    z_feature_counts = z_scores.gt(3).sum().astype(int).to_dict()

    iso_model = IsolationForest(random_state=random_state, contamination="auto")
    iso_pred = iso_model.fit_predict(numeric_X)
    iso_mask = pd.Series(iso_pred == -1, index=numeric_X.index)

    combined = (iqr_mask | z_mask | iso_mask).rename("suspected_outlier")
    report = {
        "methods": {
            "iqr": {
                "row_count": int(iqr_mask.sum()),
                "feature_counts": {key: int(value) for key, value in iqr_feature_counts.items()},
            },
            "z_score": {
                "row_count": int(z_mask.sum()),
                "feature_counts": {key: int(value) for key, value in z_feature_counts.items()},
            },
            "isolation_forest": {"row_count": int(iso_mask.sum())},
        },
        "total_suspected_outliers": int(combined.sum()),
        "sample_indices": [int(index) for index in combined.index[combined].tolist()[:30]],
        "policy": "Rows are flagged only. They are not removed automatically.",
    }
    return combined, report


def analyze_correlations(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    threshold: float = 0.95,
) -> dict[str, Any]:
    numeric_X = X.select_dtypes(include="number")
    if numeric_X.shape[1] < 2:
        return {"highly_correlated_pairs": [], "target_correlations": []}

    corr = numeric_X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs: list[dict[str, Any]] = []
    for feature_a in upper.columns:
        for feature_b, value in upper[feature_a].dropna().items():
            if value > threshold:
                pairs.append(
                    {
                        "feature_1": str(feature_b),
                        "feature_2": str(feature_a),
                        "abs_correlation": float(value),
                    }
                )

    target_correlations: list[dict[str, Any]] = []
    if y is not None and y.nunique(dropna=True) > 1:
        corr_df = numeric_X.copy()
        corr_df["target"] = y.to_numpy()
        target_corr = corr_df.corr(numeric_only=True)["target"].drop("target").sort_values(
            key=lambda series: series.abs(),
            ascending=False,
        )
        target_correlations = [
            {"feature": str(feature), "correlation_with_target": float(value)}
            for feature, value in target_corr.head(15).items()
            if pd.notna(value)
        ]

    return {
        "threshold": threshold,
        "highly_correlated_pairs": pairs,
        "target_correlations": target_correlations,
        "policy": "Highly correlated features are reported but not removed automatically.",
    }


def analyze_class_imbalance(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    ratios = {str(label): _safe_ratio(int(count), total) for label, count in counts.items()}
    minority_label = str(counts.idxmin()) if not counts.empty else None
    majority_label = str(counts.idxmax()) if not counts.empty else None
    minority_ratio = min(ratios.values()) if ratios else 0.0
    imbalance_detected = bool(minority_ratio and minority_ratio < 0.35)

    suggestions = []
    if imbalance_detected:
        suggestions.extend(
            [
                "Consider class_weight='balanced' for supported estimators.",
                "Consider SMOTE only inside a cross-validation/training pipeline to avoid leakage.",
                "Monitor Recall (M) and False Negative count instead of Accuracy only.",
            ]
        )
    else:
        suggestions.append("Class distribution is not severely imbalanced in the current dataset.")

    return {
        "counts": {str(label): int(count) for label, count in counts.items()},
        "ratios": ratios,
        "minority_class": minority_label,
        "majority_class": majority_label,
        "minority_ratio": minority_ratio,
        "imbalance_detected": imbalance_detected,
        "suggestions": suggestions,
    }


def compute_quality_score(report: dict[str, Any]) -> int:
    score = 100
    rows = report["shape"]["rows"]
    total_missing = sum(item["missing_count"] for item in report["missing_values"])
    total_cells = rows * report["shape"]["columns"]
    missing_rate = _safe_ratio(total_missing, total_cells)

    score -= min(30, int(missing_rate * 100))
    score -= min(20, report["duplicates"]["duplicate_rows"])
    score -= min(20, len(report["range_violations"]) * 5)
    score -= min(15, int(report["outliers"]["total_suspected_outliers"] / max(rows, 1) * 100))
    if report["class_balance"]["imbalance_detected"]:
        score -= 10
    return max(0, int(score))


def build_data_quality_report(
    raw_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    target_col: str,
    X: pd.DataFrame,
    y: pd.Series,
    normalization_report: dict[str, Any],
    imputation_decisions: list[dict[str, Any]],
    random_state: int,
    expected_columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.Series]:
    outlier_flags, outlier_report = detect_outliers(X, random_state=random_state)
    report = {
        "shape": {"rows": int(raw_df.shape[0]), "columns": int(raw_df.shape[1])},
        "target_column": target_col,
        "schema": validate_schema(normalized_df, target_col, expected_columns=expected_columns),
        "data_types": validate_data_types(normalized_df, target_col),
        "normalization": normalization_report,
        "missing_values": summarize_missing_values(normalized_df),
        "imputation_decisions": imputation_decisions,
        "duplicates": summarize_duplicates(normalized_df),
        "duplicate_rows": int(normalized_df.duplicated().sum()),
        "range_violations": validate_ranges(normalized_df, target_col),
        "outliers": outlier_report,
        "correlation": analyze_correlations(X, y),
        "class_balance": analyze_class_imbalance(y),
        "label_distribution": {
            str(label): int(count) for label, count in normalized_df[target_col].value_counts(dropna=False).items()
        },
        "quality_policy": {
            "missing_values": "Feature missing values are imputed by documented rules; target-missing rows are excluded because they cannot be supervised labels.",
            "outliers": "Suspected outliers are flagged, not deleted.",
            "correlations": "Highly correlated features are reported, not dropped automatically.",
        },
    }
    report["quality_score"] = compute_quality_score(report)
    return report, outlier_flags


def write_data_quality_flags(
    df: pd.DataFrame,
    outlier_flags: pd.Series,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged = df.copy()
    flagged["suspected_outlier"] = outlier_flags.reindex(flagged.index, fill_value=False).astype(bool)
    flagged.to_csv(output_path, index=False)
    return output_path


def _rows_to_html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No records.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_data_quality_html(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_rows = [
        row for row in report["missing_values"] if row["missing_count"] > 0
    ] or [{"column": "None", "missing_count": 0, "missing_percent": 0.0}]
    correlated_rows = report["correlation"]["highly_correlated_pairs"][:30]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Data Quality Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ color: #17324d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .score {{ font-size: 28px; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Data Quality Report</h1>
  <p class="score">Quality score: {report["quality_score"]}/100</p>
  <h2>Dataset Summary</h2>
  <ul>
    <li>Rows: {report["shape"]["rows"]}</li>
    <li>Columns: {report["shape"]["columns"]}</li>
    <li>Target column: {html.escape(str(report["target_column"]))}</li>
    <li>Duplicate rows: {report["duplicates"]["duplicate_rows"]}</li>
    <li>Suspected outliers: {report["outliers"]["total_suspected_outliers"]}</li>
  </ul>
  <h2>Missing Values</h2>
  {_rows_to_html_table(missing_rows, ["column", "missing_count", "missing_percent"])}
  <h2>Class Balance</h2>
  <pre>{html.escape(str(report["class_balance"]))}</pre>
  <h2>Highly Correlated Feature Pairs</h2>
  {_rows_to_html_table(correlated_rows, ["feature_1", "feature_2", "abs_correlation"])}
  <h2>Policy</h2>
  <p>Outliers and highly correlated features are reported only; they are not removed automatically.</p>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
