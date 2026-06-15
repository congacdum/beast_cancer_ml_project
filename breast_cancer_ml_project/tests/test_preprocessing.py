from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data


def _sample_dirty_dataframe() -> pd.DataFrame:
    rows = []
    labels = ["M", "B", "malignant", "benign", "M", "B", "malignant", "benign"]
    for index, label in enumerate(labels):
        rows.append(
            {
                "mean radius": 10.0 + index,
                "mean texture": 15.0 + index,
                "mean area": 250.0 + index * 5,
                "diagnosis": label,
            }
        )
    rows.append(rows[0].copy())
    rows[2]["mean texture"] = np.nan
    return pd.DataFrame(rows)


def test_missing_values_are_imputed_without_leaking_into_outputs() -> None:
    result = preprocess_data(_sample_dirty_dataframe())

    assert result["X_train"].isna().sum().sum() == 0
    assert result["X_test"].isna().sum().sum() == 0
    assert any(
        item.get("column") == "mean texture"
        for item in result["data_report"]["imputation_decisions"]
    )


def test_duplicate_rows_are_removed_and_reported() -> None:
    result = preprocess_data(_sample_dirty_dataframe())

    assert result["data_report"]["rows_after_preprocessing"] == 8
    assert any(
        item.get("strategy") == "drop_duplicate_rows"
        for item in result["data_report"]["imputation_decisions"]
    )


def test_label_normalization_maps_text_labels_to_binary_target() -> None:
    result = preprocess_data(_sample_dirty_dataframe())

    assert set(result["y"].unique()) == {0, 1}
    assert result["label_mapping"] == {0: "benign", 1: "malignant"}


def test_schema_validation_reports_target_presence() -> None:
    result = preprocess_data(_sample_dirty_dataframe())

    schema = result["data_report"]["schema"]
    assert schema["target_present"] is True
    assert schema["target_column"] == "diagnosis"
