from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from config import DATA_FILE


def load_data(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the Breast Cancer dataset from the data directory."""
    if not data_path.parent.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path.parent}. Create data/ and add the dataset CSV."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Put the CSV file in data/ first."
        )
    try:
        return pd.read_csv(data_path)
    except EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {data_path}") from exc
    except ParserError as exc:
        raise ValueError(f"Dataset file is not a valid CSV: {data_path}") from exc


def load_dataset(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """Backward-compatible alias used by the Streamlit app."""
    return load_data(data_path)
