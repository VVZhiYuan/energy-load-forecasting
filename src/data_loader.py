"""Data loading helpers for the electricity load dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DATA_FILENAME


def load_electricity_load_data(path: str | Path) -> pd.DataFrame:
    """Load the UCI electricity load dataset into a DataFrame.

    The original file is commonly stored as a semicolon-separated text file.
    Timestamp is parsed from the first column and used as the index.
    """

    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=[0],
        index_col=0,
    )
    df.index.name = "timestamp"
    df = df.sort_index()
    return df


def get_default_raw_data_path(project_root: str | Path) -> Path:
    """Return the expected raw data file path."""

    project_root = Path(project_root)
    return project_root / "data" / "raw" / RAW_DATA_FILENAME


def make_long_format(df: pd.DataFrame, value_name: str = "load") -> pd.DataFrame:
    """Convert the wide client table to long format for exploratory analysis."""

    long_df = (
        df.reset_index()
        .melt(id_vars="timestamp", var_name="meter_id", value_name=value_name)
        .dropna(subset=[value_name])
        .sort_values(["meter_id", "timestamp"])
        .reset_index(drop=True)
    )
    return long_df

