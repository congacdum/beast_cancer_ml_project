from __future__ import annotations

import pandas as pd

from src.data_validation import (
    normalize_dataframe,
    normalize_missing_tokens,
    normalize_numeric_strings,
    validate_ranges,
    validate_schema,
)


def test_missing_tokens_are_normalized_to_na() -> None:
    df = pd.DataFrame({"mean radius": ["NA", "?", "12.4"], "diagnosis": ["M", "B", "M"]})

    normalized, report = normalize_missing_tokens(df)

    assert int(normalized["mean radius"].isna().sum()) == 2
    assert report["missing_token_replacements"]["mean radius"] == 2


def test_numeric_strings_with_simple_units_are_converted() -> None:
    df = pd.DataFrame({"mean radius": ["12.4 mm", "13.1mm", "14"], "diagnosis": ["M", "B", "M"]})

    normalized, report = normalize_numeric_strings(df, skip_columns={"diagnosis"})

    assert pd.api.types.is_numeric_dtype(normalized["mean radius"])
    assert normalized["mean radius"].tolist() == [12.4, 13.1, 14.0]
    assert "mean radius" in report["numeric_string_conversions"]


def test_range_validation_flags_invalid_negative_values() -> None:
    df = pd.DataFrame({"mean radius": [12.4, -1.0], "diagnosis": ["benign", "malignant"]})

    violations = validate_ranges(df, target_col="diagnosis")

    assert "mean radius" in violations
    assert violations["mean radius"]["invalid_count"] == 1


def test_schema_validation_detects_missing_expected_columns() -> None:
    df = pd.DataFrame({"mean radius": [12.4], "diagnosis": ["benign"]})

    schema = validate_schema(
        df,
        target_col="diagnosis",
        expected_columns=["mean radius", "mean texture", "diagnosis"],
    )

    assert schema["target_present"] is True
    assert schema["missing_expected_columns"] == ["mean texture"]


def test_normalize_dataframe_normalizes_labels_and_missing_values() -> None:
    df = pd.DataFrame({"mean radius": ["?", "12.4 mm"], "diagnosis": ["M", "B"]})

    normalized, report = normalize_dataframe(df, target_col="diagnosis")

    assert int(normalized["mean radius"].isna().sum()) == 1
    assert normalized["diagnosis"].tolist() == ["malignant", "benign"]
    assert report["label_replacements"]["diagnosis"]["malignant"] == 1
