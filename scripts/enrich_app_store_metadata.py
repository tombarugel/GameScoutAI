"""
Enrich the latest App Store chart snapshot with detailed app metadata.

The script:
1. Finds the latest raw chart snapshot in data/raw.
2. Looks up detailed metadata for every App Store ID.
3. Merges the metadata with chart positions.
4. Saves the enriched dataset in data/processed.

Run from the project root:

    python scripts/enrich_app_store_metadata.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

COUNTRY = "us"

# Apple supports several IDs in one lookup request.
# Keeping batches relatively small makes debugging easier.
BATCH_SIZE = 25

REQUEST_TIMEOUT_SECONDS = 30
SECONDS_BETWEEN_REQUESTS = 1.0

LOOKUP_URL = "https://itunes.apple.com/lookup"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"


# =============================================================================
# FILE UTILITIES
# =============================================================================

def find_latest_raw_snapshot() -> Path:
    """
    Find the most recently modified App Store chart CSV.

    Returns:
        Path to the latest raw snapshot.

    Raises:
        FileNotFoundError:
            If data/raw contains no CSV file.
    """
    csv_files = list(RAW_DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No raw App Store CSV was found in data/raw."
        )

    return max(
        csv_files,
        key=lambda path: path.stat().st_mtime,
    )


def load_raw_snapshot(snapshot_path: Path) -> pd.DataFrame:
    """
    Load and validate the raw chart snapshot.
    """
    dataframe = pd.read_csv(snapshot_path)

    required_columns = {
        "app_id",
        "title",
        "developer",
        "chart_type",
        "chart_position",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            "The raw snapshot is missing required columns: "
            f"{missing_text}"
        )

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    dataframe = dataframe.drop_duplicates(
        subset=["app_id", "chart_type"]
    ).copy()

    return dataframe


# =============================================================================
# LOOKUP API
# =============================================================================

def split_into_batches(
    values: list[str],
    batch_size: int,
) -> list[list[str]]:
    """
    Split a list into smaller batches.

    Example:
        [1, 2, 3, 4, 5] with batch_size=2
        becomes [[1, 2], [3, 4], [5]].
    """
    return [
        values[index:index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def fetch_lookup_batch(
    app_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Retrieve detailed metadata for one batch of App Store IDs.

    Args:
        app_ids:
            App Store identifiers to request.

    Returns:
        List of app metadata dictionaries returned by Apple.

    Raises:
        requests.RequestException:
            If the HTTP request fails.
        ValueError:
            If Apple returns invalid JSON.
    """
    params = {
        "id": ",".join(app_ids),
        "country": COUNTRY,
        "entity": "software",
    }

    response = requests.get(
        LOOKUP_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": (
                "GameScoutAI/1.0 "
                "(educational App Store analysis project)"
            )
        },
    )

    response.raise_for_status()

    try:
        payload = response.json()

    except json.JSONDecodeError as error:
        raise ValueError(
            "Apple returned an invalid JSON response."
        ) from error

    results = payload.get("results", [])

    if not isinstance(results, list):
        raise ValueError(
            "Unexpected format in the Apple lookup response."
        )

    return results


def collect_metadata(
    app_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Collect metadata for all App Store IDs.

    Failed batches are reported and skipped so that one request
    does not necessarily stop the entire enrichment process.
    """
    batches = split_into_batches(
        values=app_ids,
        batch_size=BATCH_SIZE,
    )

    all_results: list[dict[str, Any]] = []

    for batch_number, batch in enumerate(batches, start=1):
        print(
            f"Fetching batch {batch_number}/{len(batches)} "
            f"({len(batch)} app IDs)..."
        )

        try:
            batch_results = fetch_lookup_batch(batch)

        except (requests.RequestException, ValueError) as error:
            print(
                f"Warning: batch {batch_number} failed: {error}"
            )
            continue

        all_results.extend(batch_results)

        print(
            f"Received {len(batch_results)} app records."
        )

        if batch_number < len(batches):
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    return all_results


# =============================================================================
# METADATA TRANSFORMATION
# =============================================================================

def parse_metadata(
    lookup_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert Apple's lookup response into a clean DataFrame.
    """
    records: list[dict[str, Any]] = []

    for app in lookup_results:
        genres = app.get("genres", [])

        if not isinstance(genres, list):
            genres = []

        record = {
            "app_id": str(app.get("trackId", "")),
            "lookup_title": app.get("trackName"),
            "lookup_developer": app.get("artistName"),
            "description": app.get("description"),
            "primary_genre": app.get("primaryGenreName"),
            "detailed_genres": " | ".join(genres),
            "average_rating": app.get(
                "averageUserRating"
            ),
            "rating_count": app.get(
                "userRatingCount"
            ),
            "current_version_rating": app.get(
                "averageUserRatingForCurrentVersion"
            ),
            "current_version_rating_count": app.get(
                "userRatingCountForCurrentVersion"
            ),
            "price": app.get("price"),
            "currency": app.get("currency"),
            "formatted_price": app.get(
                "formattedPrice"
            ),
            "release_date": app.get("releaseDate"),
            "last_update_date": app.get(
                "currentVersionReleaseDate"
            ),
            "version": app.get("version"),
            "minimum_os_version": app.get(
                "minimumOsVersion"
            ),
            "content_rating": app.get(
                "contentAdvisoryRating"
            ),
            "seller_name": app.get("sellerName"),
            "bundle_id": app.get("bundleId"),
            "file_size_bytes": app.get(
                "fileSizeBytes"
            ),
            "supported_devices": " | ".join(
                app.get("supportedDevices", [])
            ),
            "language_codes": " | ".join(
                app.get("languageCodesISO2A", [])
            ),
            "artwork_url_512": app.get(
                "artworkUrl512"
            ),
            "lookup_app_store_url": app.get(
                "trackViewUrl"
            ),
            "has_in_app_purchases": bool(
                app.get("features")
            ),
        }

        records.append(record)

    metadata_dataframe = pd.DataFrame(records)

    if metadata_dataframe.empty:
        return metadata_dataframe

    metadata_dataframe = metadata_dataframe.drop_duplicates(
        subset=["app_id"]
    )

    numeric_columns = [
        "average_rating",
        "rating_count",
        "current_version_rating",
        "current_version_rating_count",
        "price",
        "file_size_bytes",
    ]

    for column in numeric_columns:
        metadata_dataframe[column] = pd.to_numeric(
            metadata_dataframe[column],
            errors="coerce",
        )

    date_columns = [
        "release_date",
        "last_update_date",
    ]

    for column in date_columns:
        metadata_dataframe[column] = pd.to_datetime(
            metadata_dataframe[column],
            errors="coerce",
            utc=True,
        )

    return metadata_dataframe


# =============================================================================
# MERGING AND SAVING
# =============================================================================

def merge_chart_and_metadata(
    raw_dataframe: pd.DataFrame,
    metadata_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge chart rankings with detailed Apple metadata.
    """
    if metadata_dataframe.empty:
        raise RuntimeError(
            "No metadata was retrieved from Apple."
        )

    enriched_dataframe = raw_dataframe.merge(
        metadata_dataframe,
        on="app_id",
        how="left",
        validate="many_to_one",
    )

    enriched_dataframe["metadata_found"] = (
        enriched_dataframe["description"].notna()
    )

    return enriched_dataframe


def save_enriched_data(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Save the enriched dataset to data/processed.
    """
    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIRECTORY
        / "app_store_games_enriched.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def display_summary(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print a short quality summary to the terminal.
    """
    total_rows = len(dataframe)

    found_rows = int(
        dataframe["metadata_found"].sum()
    )

    missing_rows = total_rows - found_rows

    print("\nEnrichment summary")
    print("------------------")
    print(f"Rows: {total_rows}")
    print(f"Metadata found: {found_rows}")
    print(f"Metadata missing: {missing_rows}")

    if "average_rating" in dataframe.columns:
        ratings_available = int(
            dataframe["average_rating"]
            .notna()
            .sum()
        )

        print(
            f"Ratings available: {ratings_available}"
        )

    if "description" in dataframe.columns:
        descriptions_available = int(
            dataframe["description"]
            .notna()
            .sum()
        )

        print(
            "Descriptions available: "
            f"{descriptions_available}"
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Run the complete metadata enrichment pipeline.
    """
    print("Finding latest raw App Store snapshot...")

    raw_snapshot_path = find_latest_raw_snapshot()

    print(f"Using: {raw_snapshot_path.name}")

    raw_dataframe = load_raw_snapshot(
        raw_snapshot_path
    )

    unique_app_ids = (
        raw_dataframe["app_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique App Store IDs to enrich: "
        f"{len(unique_app_ids)}"
    )

    lookup_results = collect_metadata(
        unique_app_ids
    )

    print(
        f"\nTotal metadata records received: "
        f"{len(lookup_results)}"
    )

    metadata_dataframe = parse_metadata(
        lookup_results
    )

    enriched_dataframe = merge_chart_and_metadata(
        raw_dataframe=raw_dataframe,
        metadata_dataframe=metadata_dataframe,
    )

    output_path = save_enriched_data(
        enriched_dataframe
    )

    display_summary(enriched_dataframe)

    print("\nEnrichment completed successfully.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()