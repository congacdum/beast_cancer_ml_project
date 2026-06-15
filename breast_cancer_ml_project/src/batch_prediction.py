from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_FILE, DEMO_PATIENTS_FILE, PREDICTION_TEMPLATE_FILE, RANDOM_STATE
from src.data_utils import load_data
from src.data_validation import (
    detect_outliers,
    normalize_dataframe,
    summarize_duplicates,
    summarize_missing_values,
    validate_ranges,
    validate_schema,
)
from src.predict import load_decision_threshold
from src.preprocessing import detect_target_column


def load_prediction_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.casefold()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV and XLSX files are supported.")


def get_reference_feature_frame(feature_names: list[str]) -> pd.DataFrame | None:
    if not DATA_FILE.exists():
        return None
    df = load_data(DATA_FILE)
    return df.reindex(columns=feature_names)


def get_sample_row(feature_names: list[str]) -> pd.DataFrame:
    reference = get_reference_feature_frame(feature_names)
    if reference is None or reference.empty:
        return pd.DataFrame([[0.0 for _ in feature_names]], columns=feature_names)
    return reference.head(1).reset_index(drop=True)


def create_prediction_template(feature_names: list[str]) -> bytes:
    sample_row = get_sample_row(feature_names)
    instructions = pd.DataFrame(
        {
            "Instruction": [
                "Fill one patient per row in the prediction_template sheet.",
                "Do not rename, remove, or reorder required feature columns.",
                "CSV and XLSX uploads are supported.",
                "Predictions are for academic/demo use only and do not replace medical diagnosis.",
            ]
        }
    )
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample_row.to_excel(writer, sheet_name="prediction_template", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    return output.getvalue()


def save_prediction_template(feature_names: list[str], output_path: Path = PREDICTION_TEMPLATE_FILE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(create_prediction_template(feature_names))
    return output_path


def create_demo_patients(feature_names: list[str]) -> pd.DataFrame:
    df = load_data(DATA_FILE)
    target_col = detect_target_column(df)
    benign = df[df[target_col].astype(str).str.casefold().eq("benign")].head(2)
    malignant = df[df[target_col].astype(str).str.casefold().eq("malignant")].head(2)
    demo = pd.concat([benign, malignant], axis=0).reset_index(drop=True)
    demo.insert(
        0,
        "case_name",
        ["Benign Sample A", "Benign Sample B", "Malignant Sample A", "Malignant Sample B"],
    )
    return demo[["case_name", target_col] + feature_names]


def save_demo_patients(feature_names: list[str], output_path: Path = DEMO_PATIENTS_FILE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo = create_demo_patients(feature_names)
    instructions = pd.DataFrame(
        {
            "Instruction": [
                "These demo patients are sampled from the real dataset for presentation use.",
                "Use case_name to select a sample in Streamlit.",
                "diagnosis is included for demo reference and is not required for prediction uploads.",
            ]
        }
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        demo.to_excel(writer, sheet_name="demo_patients", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    return output_path


def load_demo_patients(feature_names: list[str]) -> pd.DataFrame:
    if not DEMO_PATIENTS_FILE.exists():
        save_demo_patients(feature_names)
    return pd.read_excel(DEMO_PATIENTS_FILE, sheet_name="demo_patients")


def find_unusual_values(
    input_df: pd.DataFrame,
    feature_names: list[str],
    reference_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if reference_df is None:
        reference_df = get_reference_feature_frame(feature_names)
    if reference_df is None or reference_df.empty:
        return []

    warnings: list[dict[str, Any]] = []
    for feature in feature_names:
        if feature not in input_df.columns or feature not in reference_df.columns:
            continue
        ref = pd.to_numeric(reference_df[feature], errors="coerce").dropna()
        if ref.empty:
            continue
        q1 = ref.quantile(0.25)
        q3 = ref.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr if iqr else ref.min()
        upper = q3 + 3 * iqr if iqr else ref.max()
        values = pd.to_numeric(input_df[feature], errors="coerce")
        mask = values.notna() & ~values.between(lower, upper)
        for index, value in values[mask].items():
            warnings.append(
                {
                    "row": int(index),
                    "column": feature,
                    "type": "medical_validation_warning",
                    "value": float(value),
                    "expected_range": f"{lower:.6g} to {upper:.6g}",
                    "message": "Value is outside the usual training-data range. Please check again.",
                }
            )
    return warnings


def validate_prediction_batch(
    input_df: pd.DataFrame,
    feature_names: list[str],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized, normalization_report = normalize_dataframe(input_df, target_col=None)
    schema = validate_schema(normalized, target_col="", expected_columns=feature_names)
    missing_required = [feature for feature in feature_names if feature not in normalized.columns]
    extra_columns = [column for column in normalized.columns if column not in feature_names]

    invalid_cells: list[dict[str, Any]] = []
    for feature in missing_required:
        invalid_cells.append(
            {
                "row": None,
                "column": feature,
                "type": "missing_required_column",
                "message": "Required feature column is missing.",
            }
        )

    available_features = [feature for feature in feature_names if feature in normalized.columns]
    feature_df = normalized.reindex(columns=feature_names)
    numeric_df = feature_df.copy()
    for feature in available_features:
        numeric_values = pd.to_numeric(numeric_df[feature], errors="coerce")
        invalid_mask = numeric_df[feature].notna() & numeric_values.isna()
        for index, value in numeric_df.loc[invalid_mask, feature].items():
            invalid_cells.append(
                {
                    "row": int(index),
                    "column": feature,
                    "type": "invalid_numeric_value",
                    "value": str(value),
                    "message": "Feature value must be numeric.",
                }
            )
        numeric_df[feature] = numeric_values

    missing_rows = summarize_missing_values(numeric_df)
    for item in missing_rows:
        if item["missing_count"] > 0:
            missing_indices = numeric_df.index[numeric_df[item["column"]].isna()].tolist()[:20]
            for index in missing_indices:
                invalid_cells.append(
                    {
                        "row": int(index),
                        "column": item["column"],
                        "type": "missing_value",
                        "message": "Missing value must be fixed before batch prediction.",
                    }
                )

    range_violations = validate_ranges(numeric_df, target_col="")
    for column, details in range_violations.items():
        for index in details.get("sample_indices", []):
            invalid_cells.append(
                {
                    "row": int(index),
                    "column": column,
                    "type": "range_violation",
                    "message": details["rule"],
                }
            )

    if len(numeric_df.dropna()) >= 2 and not missing_required:
        outlier_flags, outlier_report = detect_outliers(numeric_df, random_state=random_state)
    else:
        outlier_flags = pd.Series(False, index=numeric_df.index, name="suspected_outlier")
        outlier_report = {
            "methods": {},
            "total_suspected_outliers": 0,
            "policy": "Outlier detection skipped for fewer than 2 valid rows.",
        }

    unusual_values = find_unusual_values(numeric_df, feature_names)
    report = {
        "records_loaded": int(input_df.shape[0]),
        "required_columns": feature_names,
        "missing_required_columns": missing_required,
        "extra_columns": extra_columns,
        "schema": schema,
        "normalization": normalization_report,
        "missing_values": missing_rows,
        "duplicate_rows": summarize_duplicates(normalized),
        "range_violations": range_violations,
        "outliers": outlier_report,
        "unusual_values": unusual_values,
        "invalid_cells": invalid_cells,
        "can_predict": len(invalid_cells) == 0,
    }
    numeric_df["suspected_outlier"] = outlier_flags.reindex(numeric_df.index, fill_value=False).astype(bool)
    return numeric_df, report


def risk_level(probability_malignant: float | None, prediction_code: int) -> str:
    if probability_malignant is None:
        return "High" if prediction_code == 1 else "Low"
    if probability_malignant >= 0.70:
        return "High"
    if probability_malignant >= 0.40:
        return "Medium"
    return "Low"


def predict_batch_dataframe(
    validated_df: pd.DataFrame,
    model,
    feature_names: list[str],
    label_mapping: dict[int, str],
) -> pd.DataFrame:
    feature_df = validated_df.loc[:, feature_names].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(feature_df.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must not contain NaN or infinite values.")

    decision_threshold = load_decision_threshold()
    prediction_codes = model.predict(feature_df).astype(int)
    probability_benign: np.ndarray | list[float | None] = [None] * len(feature_df)
    probability_malignant: np.ndarray | list[float | None] = [None] * len(feature_df)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_df)
        probability_benign = probabilities[:, 0]
        probability_malignant = probabilities[:, 1]
        prediction_codes = (probability_malignant >= decision_threshold).astype(int)

    result_df = pd.DataFrame(
        {
            "row": [int(index) for index in feature_df.index],
            "prediction_code": prediction_codes,
            "prediction": [
                label_mapping.get(int(code), str(int(code))) for code in prediction_codes
            ],
            "probability_benign": probability_benign,
            "probability_malignant": probability_malignant,
            "decision_threshold": decision_threshold,
            "suspected_outlier": validated_df["suspected_outlier"].astype(bool).to_numpy(),
        }
    )
    result_df["risk_level"] = [
        risk_level(probability, int(code))
        for probability, code in zip(
            result_df["probability_malignant"],
            result_df["prediction_code"],
        )
    ]
    return pd.concat([feature_df.reset_index(drop=True), result_df], axis=1)


def export_prediction_results(results_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="prediction_results", index=False)
    return output.getvalue()
