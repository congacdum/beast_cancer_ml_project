from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CV_RESULTS_FILE,
    DATA_FILE,
    DATA_QUALITY_REPORT_FILE,
    FIGURE_DIR,
    MODEL_COMPARISON_FILE,
    MODEL_DIR,
    ROOT_DIR,
    TARGET_COL,
    THRESHOLD_RESULTS_FILE,
    TRAINING_METADATA_FILE,
    TUNING_RESULTS_FILE,
)
from src.data_utils import load_data
from src.predict import load_model_assets
from src.preprocessing import detect_target_column, inspect_data
from src.visualization import model_slug


def load_data_safe() -> tuple[pd.DataFrame | None, str | None]:
    try:
        return load_data(DATA_FILE), None
    except FileNotFoundError:
        return None, f"Không tìm thấy dữ liệu tại: {DATA_FILE.relative_to(ROOT_DIR)}"
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Không thể đọc dữ liệu: {exc}"


def load_model_safe():
    try:
        model, feature_names, label_mapping = load_model_assets()
        return (model, feature_names, label_mapping), None
    except FileNotFoundError:
        return None, "Chưa có model. Hãy train model trước ở trang Huấn luyện mô hình."
    except Exception as exc:
        return None, f"Không thể load model: {exc}"


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_table(path: Path, index_col: int | str | None = 0) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def load_results_table() -> pd.DataFrame | None:
    return load_csv_table(MODEL_COMPARISON_FILE, index_col=0)


def load_cv_results_table() -> pd.DataFrame | None:
    return load_csv_table(CV_RESULTS_FILE, index_col=0)


def load_tuning_results_table() -> pd.DataFrame | None:
    return load_csv_table(TUNING_RESULTS_FILE, index_col=0)


def load_threshold_results_table() -> pd.DataFrame | None:
    return load_csv_table(THRESHOLD_RESULTS_FILE, index_col=None)


def get_data_profile(df: pd.DataFrame) -> dict[str, Any]:
    target_col = detect_target_column(df, TARGET_COL)
    return inspect_data(df, target_col)


def get_feature_defaults(
    df: pd.DataFrame | None,
    feature_names: list[str],
) -> dict[str, float]:
    if df is None:
        return {feature: 0.0 for feature in feature_names}

    numeric_medians = df.reindex(columns=feature_names).median(numeric_only=True)
    defaults = {}
    for feature in feature_names:
        value = numeric_medians.get(feature, 0.0)
        defaults[feature] = 0.0 if pd.isna(value) else float(value)
    return defaults


def make_target_distribution_figure(df: pd.DataFrame, target_col: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def make_correlation_heatmap_figure(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def get_saved_model_names() -> list[str]:
    if not MODEL_DIR.exists():
        return []
    ignored = {
        "best_model.joblib",
        "feature_names.joblib",
        "label_mapping.joblib",
        "decision_threshold.joblib",
    }
    return sorted(path.stem for path in MODEL_DIR.glob("*.joblib") if path.name not in ignored)


def get_model_figure_path(model_name: str, figure_type: str) -> Path:
    return FIGURE_DIR / f"{figure_type}_{model_slug(model_name)}.png"


def load_training_metadata() -> dict[str, Any] | None:
    return load_json_file(TRAINING_METADATA_FILE)


def load_data_quality_report() -> dict[str, Any] | None:
    return load_json_file(DATA_QUALITY_REPORT_FILE)
