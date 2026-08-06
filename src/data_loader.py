"""Data loading helpers for the electricity load dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA_FILENAME


@dataclass(frozen=True)
class LoadedLoadSeries:
    series: pd.Series
    input_format: str
    source_label: str
    meter: str | None
    negative_load_count: int


def _read_forecast_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        sample = stream.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error as exc:
        raise ValueError(f"Could not detect a supported delimiter in {path}.") from exc
    decimal = "," if delimiter == ";" else "."
    try:
        return pd.read_csv(path, sep=delimiter, decimal=decimal)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Forecast input is empty: {path}") from exc


def _validate_normalized_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.hasnans:
        raise ValueError("timestamps must be parseable and present.")
    series = series.sort_index()
    if not series.index.is_unique:
        raise ValueError("timestamps must be unique.")
    expected = pd.date_range(series.index[0], series.index[-1], freq="15min")
    if not series.index.equals(expected):
        raise ValueError("timestamps must have a continuous 15-minute frequency.")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("load must contain only finite numeric values.")
    numeric.name = "load"
    return numeric


def load_forecast_series(path: str | Path, meter: str | None = None) -> LoadedLoadSeries:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Forecast input file not found: {path}")
    frame = _read_forecast_table(path)
    serialized_index_columns = []
    for column in frame.columns:
        if not str(column).startswith("Unnamed:"):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().all() and np.array_equal(numeric.to_numpy(), np.arange(len(frame))):
            serialized_index_columns.append(column)
    frame = frame.drop(columns=serialized_index_columns)
    if frame.empty:
        raise ValueError(f"Forecast input is empty: {path}")
    names = {str(column).strip().lower(): column for column in frame.columns}

    if {"timestamp", "load"}.issubset(names) and set(names) != {"timestamp", "load"}:
        raise ValueError("ambiguous input: timestamp,load schema cannot contain extra columns.")
    if set(names) == {"timestamp", "load"}:
        timestamp_column = names["timestamp"]
        value_column = names["load"]
        input_format = "long"
        selected_meter = None
    else:
        if frame.shape[1] < 2:
            raise ValueError("Input must be timestamp,load or a timestamp plus meter columns.")
        timestamp_column = frame.columns[0]
        if meter is None:
            raise ValueError("--meter is required for wide input.")
        if meter not in frame.columns:
            raise ValueError(f"Meter {meter!r} was not found in the input columns.")
        value_column = meter
        input_format = "wide"
        selected_meter = meter

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    series = pd.Series(frame[value_column].to_numpy(), index=timestamps, name="load")
    series = _validate_normalized_series(series)
    return LoadedLoadSeries(
        series=series,
        input_format=input_format,
        source_label=selected_meter or path.stem,
        meter=selected_meter,
        negative_load_count=int((series < 0).sum()),
    )


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
