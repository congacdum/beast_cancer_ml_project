from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import POSITIVE_CLASS, RANDOM_STATE, TARGET_CANDIDATES, TARGET_COL, TEST_SIZE
from src.data_validation import build_data_quality_report, normalize_dataframe


def detect_target_column(df: pd.DataFrame, target_col: str | None = TARGET_COL) -> str:
    if target_col and target_col in df.columns:
        return target_col

    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Target column could not be detected. Set TARGET_COL in config.py "
        "or rename the target column to one of: "
        f"{', '.join(TARGET_CANDIDATES)}."
    )


def inspect_data(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": target_col,
        "label_distribution": {
            str(label): int(count)
            for label, count in df[target_col].value_counts(dropna=False).items()
        },
    }


def _mode_or_unknown(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else "unknown"


def _median_or_zero(series: pd.Series) -> float:
    median = pd.to_numeric(series, errors="coerce").median()
    return 0.0 if pd.isna(median) else float(median)


def _missing_decision(
    column: str,
    train_missing: int,
    test_missing: int,
    train_total: int,
    strategy: str,
    reason: str,
    suggested_drop: bool = False,
) -> dict[str, Any]:
    train_percent = float(train_missing / train_total) if train_total else 0.0
    return {
        "column": column,
        "train_missing_count": int(train_missing),
        "test_missing_count": int(test_missing),
        "train_missing_percent": train_percent,
        "strategy": strategy,
        "reason": reason,
        "suggested_drop": bool(suggested_drop),
    }


def handle_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Impute features using rules fitted only on the train set."""
    train_clean = X_train.copy()
    test_clean = X_test.copy()
    decisions: list[dict[str, Any]] = []
    knn_columns: list[str] = []
    train_total = int(train_clean.shape[0])
    numeric_columns = train_clean.select_dtypes(include="number").columns.tolist()

    for column in train_clean.columns:
        train_missing = int(train_clean[column].isna().sum())
        test_missing = int(test_clean[column].isna().sum()) if column in test_clean else 0
        if train_missing == 0 and test_missing == 0:
            continue

        train_percent = float(train_missing / train_total) if train_total else 0.0
        is_numeric = column in numeric_columns

        if train_percent < 0.05:
            if is_numeric:
                fill_value = _median_or_zero(train_clean[column])
                train_clean[column] = train_clean[column].fillna(fill_value)
                test_clean[column] = test_clean[column].fillna(fill_value)
                strategy = "median"
            else:
                fill_value = _mode_or_unknown(train_clean[column])
                train_clean[column] = train_clean[column].fillna(fill_value)
                test_clean[column] = test_clean[column].fillna(fill_value)
                strategy = "mode"
            decisions.append(
                _missing_decision(
                    column,
                    train_missing,
                    test_missing,
                    train_total,
                    strategy,
                    "Missing rate below 5%; imputed from train-set statistic.",
                )
            )
        elif 0.05 <= train_percent <= 0.30 and is_numeric:
            knn_columns.append(column)
            decisions.append(
                _missing_decision(
                    column,
                    train_missing,
                    test_missing,
                    train_total,
                    "KNNImputer",
                    "Missing rate between 5% and 30%; KNN imputer fitted on train set.",
                )
            )
        elif train_percent > 0.70:
            if is_numeric:
                fill_value = _median_or_zero(train_clean[column])
                strategy = "median_with_drop_recommendation"
            else:
                fill_value = _mode_or_unknown(train_clean[column])
                strategy = "mode_with_drop_recommendation"
            train_clean[column] = train_clean[column].fillna(fill_value)
            test_clean[column] = test_clean[column].fillna(fill_value)
            decisions.append(
                _missing_decision(
                    column,
                    train_missing,
                    test_missing,
                    train_total,
                    strategy,
                    "Missing rate above 70%; column should be reviewed before production use.",
                    suggested_drop=True,
                )
            )
        else:
            if is_numeric:
                fill_value = _median_or_zero(train_clean[column])
                strategy = "median_high_missing"
            else:
                fill_value = _mode_or_unknown(train_clean[column])
                strategy = "mode_high_missing"
            train_clean[column] = train_clean[column].fillna(fill_value)
            test_clean[column] = test_clean[column].fillna(fill_value)
            decisions.append(
                _missing_decision(
                    column,
                    train_missing,
                    test_missing,
                    train_total,
                    strategy,
                    "Missing rate above 30%; imputed for continuity and reported for review.",
                )
            )

    if knn_columns:
        numeric_columns = train_clean.select_dtypes(include="number").columns.tolist()
        imputer = KNNImputer(n_neighbors=5)
        train_numeric = pd.DataFrame(
            imputer.fit_transform(train_clean[numeric_columns]),
            columns=numeric_columns,
            index=train_clean.index,
        )
        test_numeric = pd.DataFrame(
            imputer.transform(test_clean[numeric_columns]),
            columns=numeric_columns,
            index=test_clean.index,
        )
        for column in knn_columns:
            train_clean[column] = train_numeric[column]
            test_clean[column] = test_numeric[column]

    return train_clean, test_clean, decisions


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(X_train, drop_first=False)
    test_encoded = pd.get_dummies(X_test, drop_first=False)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_encoded, test_encoded


def encode_target(y: pd.Series) -> tuple[pd.Series, dict[int, str]]:
    unique_labels = list(pd.Series(y).dropna().unique())

    if pd.api.types.is_numeric_dtype(y) and set(unique_labels).issubset({0, 1}):
        encoded = y.astype(int)
        return encoded, {int(label): str(label) for label in sorted(unique_labels)}

    if POSITIVE_CLASS in unique_labels and len(unique_labels) == 2:
        negative_label = next(label for label in unique_labels if label != POSITIVE_CLASS)
        encoded = y.map({negative_label: 0, POSITIVE_CLASS: 1}).astype(int)
        return encoded, {0: str(negative_label), 1: POSITIVE_CLASS}

    sorted_labels = sorted(str(label) for label in unique_labels)
    label_to_int = {label: index for index, label in enumerate(sorted_labels)}
    encoded = y.astype(str).map(label_to_int).astype(int)
    int_to_label = {index: label for label, index in label_to_int.items()}
    return encoded, int_to_label


def preprocess_data(
    df: pd.DataFrame,
    target_col: str | None = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    target_col = detect_target_column(df, target_col)
    normalized, normalization_report = normalize_dataframe(df, target_col=target_col)
    data_report = inspect_data(normalized, target_col)

    cleaned = normalized.copy()
    target_missing = int(cleaned[target_col].isna().sum())
    preprocessing_decisions: list[dict[str, Any]] = []
    if target_missing:
        cleaned = cleaned.dropna(subset=[target_col]).copy()
        preprocessing_decisions.append(
            {
                "column": target_col,
                "strategy": "drop_rows",
                "affected_rows": target_missing,
                "reason": "Rows without target labels cannot be used for supervised training.",
            }
        )

    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().copy()
        preprocessing_decisions.append(
            {
                "strategy": "drop_duplicate_rows",
                "affected_rows": duplicate_rows,
                "reason": "Duplicate rows can leak identical samples across train/test split.",
            }
        )

    cleaned = cleaned.reset_index(drop=True)
    X_raw = cleaned.drop(columns=[target_col])
    y_raw = cleaned[target_col]
    y, label_mapping = encode_target(y_raw)

    stratify = y if y.nunique() > 1 else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    X_train_imputed, X_test_imputed, imputation_decisions = handle_missing_values(
        X_train_raw,
        X_test_raw,
    )
    X_train, X_test = encode_features(X_train_imputed, X_test_imputed)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_all = pd.concat([X_train, X_test], axis=0).sort_index()
    y_all = pd.concat([y_train, y_test], axis=0).sort_index()
    all_decisions = preprocessing_decisions + imputation_decisions
    quality_report, outlier_flags = build_data_quality_report(
        raw_df=df,
        normalized_df=cleaned,
        target_col=target_col,
        X=X_all,
        y=y_all,
        normalization_report=normalization_report,
        imputation_decisions=all_decisions,
        random_state=random_state,
        expected_columns=list(df.columns),
    )

    data_report.update(quality_report)
    data_report["rows_after_preprocessing"] = int(cleaned.shape[0])
    data_report["feature_count_after_encoding"] = int(X_train.shape[1])
    data_report["test_size"] = test_size
    data_report["random_state"] = random_state
    data_report["stratify"] = bool(stratify is not None)

    return {
        "X": X_all,
        "y": y_all,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
        "target_col": target_col,
        "label_mapping": label_mapping,
        "cleaned_df": cleaned,
        "outlier_flags": outlier_flags,
        "data_report": data_report,
    }
