from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BEST_MODEL_FILE, DECISION_THRESHOLD_FILE, FEATURE_NAMES_FILE, LABEL_MAPPING_FILE


def load_model_assets(
    model_path: Path = BEST_MODEL_FILE,
    feature_names_path: Path = FEATURE_NAMES_FILE,
    label_mapping_path: Path = LABEL_MAPPING_FILE,
):
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    try:
        label_mapping = joblib.load(label_mapping_path)
    except FileNotFoundError:
        label_mapping = {0: "benign", 1: "malignant"}
    return model, feature_names, label_mapping


def load_decision_threshold(default: float = 0.5) -> float:
    try:
        threshold = joblib.load(DECISION_THRESHOLD_FILE)
        return float(threshold)
    except FileNotFoundError:
        return default


def predict_dataframe(
    input_df: pd.DataFrame,
    model=None,
    feature_names: list[str] | None = None,
    label_mapping: dict[int, str] | None = None,
    decision_threshold: float | None = None,
) -> dict[str, object]:
    if model is None or feature_names is None:
        model, feature_names, label_mapping = load_model_assets()

    label_mapping = label_mapping or {0: "benign", 1: "malignant"}
    missing = [feature for feature in feature_names if feature not in input_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = input_df.loc[:, feature_names].copy()
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError("All feature values must be numeric.") from exc

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must not contain NaN or infinite values.")

    decision_threshold = load_decision_threshold() if decision_threshold is None else decision_threshold
    prediction_code = int(model.predict(X)[0])
    probability_benign = None
    probability_malignant = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability_benign = float(proba[0])
        probability_malignant = float(proba[1])
        prediction_code = int(probability_malignant >= decision_threshold)

    result = {
        "prediction_code": prediction_code,
        "prediction": label_mapping.get(prediction_code, str(prediction_code)),
        "decision_threshold": float(decision_threshold),
    }

    if probability_benign is not None and probability_malignant is not None:
        result["probability_benign"] = probability_benign
        result["probability_malignant"] = probability_malignant

    return result


def predict_from_json(input_json: str):
    model, feature_names, label_mapping = load_model_assets()
    values = json.loads(input_json)
    missing = [feature for feature in feature_names if feature not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    input_df = pd.DataFrame([[values[feature] for feature in feature_names]], columns=feature_names)
    return predict_dataframe(input_df, model, feature_names, label_mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict breast cancer diagnosis from JSON feature values.")
    parser.add_argument("--json", required=True, help="JSON string with all feature values.")
    args = parser.parse_args()
    print(json.dumps(predict_from_json(args.json), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
