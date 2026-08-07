"""
Utilities used to load the latest App Store snapshot.

The application automatically selects the most recently modified CSV file
stored in data/raw. This allows the data collection script to generate
timestamped snapshots without requiring changes in the Streamlit application.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


def find_latest_snapshot() -> Path:
    """
    Return the most recently modified CSV snapshot.

    Raises:
        FileNotFoundError:
            If no CSV file exists in data/raw.
    """
    csv_files = list(RAW_DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No App Store snapshot was found in data/raw."
        )

    return max(
        csv_files,
        key=lambda file_path: file_path.stat().st_mtime,
    )


def load_latest_snapshot() -> tuple[pd.DataFrame, Path]:
    """
    Load and lightly validate the latest App Store snapshot.

    Returns:
        A tuple containing:
        - the cleaned DataFrame;
        - the path of the snapshot being used.
    """
    snapshot_path = find_latest_snapshot()

    dataframe = pd.read_csv(snapshot_path)

    required_columns = {
        "app_id",
        "title",
        "developer",
        "chart_type",
        "chart_position",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            "The latest snapshot is missing required columns: "
            f"{missing_text}"
        )

    dataframe["chart_position"] = pd.to_numeric(
        dataframe["chart_position"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "app_id",
            "title",
            "chart_type",
            "chart_position",
        ]
    ).copy()

    dataframe["app_id"] = dataframe["app_id"].astype(str)
    dataframe["chart_position"] = dataframe[
        "chart_position"
    ].astype(int)

    dataframe = dataframe.sort_values(
        by=["chart_type", "chart_position"]
    ).reset_index(drop=True)

    return dataframe, snapshot_path