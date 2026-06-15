from __future__ import annotations

import pandas as pd

from config import BEST_MODEL_FILE, DATA_FILE, FEATURE_NAMES_FILE
from src.predict import load_model_assets, predict_dataframe


def test_model_assets_load_from_disk() -> None:
    assert BEST_MODEL_FILE.exists(), "Run python src/train.py before pytest to create model artifacts."
    assert FEATURE_NAMES_FILE.exists(), "Run python src/train.py before pytest to create feature names."

    model, feature_names, label_mapping = load_model_assets()

    assert model is not None
    assert len(feature_names) > 0
    assert 0 in label_mapping and 1 in label_mapping


def test_prediction_output_and_probability_range() -> None:
    model, feature_names, label_mapping = load_model_assets()
    df = pd.read_csv(DATA_FILE)
    input_df = df.loc[[0], feature_names]

    result = predict_dataframe(input_df, model, feature_names, label_mapping)

    assert result["prediction_code"] in {0, 1}
    assert result["prediction"] in set(label_mapping.values())
    assert "probability_benign" in result
    assert "probability_malignant" in result
    assert 0 <= result["probability_benign"] <= 1
    assert 0 <= result["probability_malignant"] <= 1
    assert abs(result["probability_benign"] + result["probability_malignant"] - 1) < 1e-6
