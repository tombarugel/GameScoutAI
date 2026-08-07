"""
Collect current US App Store game rankings.

This first version retrieves Apple's public chart feeds and stores the raw
ranking data as CSV files. Metadata enrichment and review collection will be
added in later steps.

Run from the project root with:

    python scripts/collect_app_store_data.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COUNTRY = "us"
CHART_LIMIT = 100
REQUEST_TIMEOUT_SECONDS = 30

BASE_FEED_URL = "https://itunes.apple.com"

CHARTS = {
    "top_free": "topfreeapplications",
    "top_paid": "toppaidapplications",
}

# Absolute project paths, independent of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_feed_url(chart_slug: str) -> str:
    """
    Build Apple's legacy RSS URL for the US Games category.

    Genre 6014 corresponds to Games.
    """
    return (
        f"{BASE_FEED_URL}/{COUNTRY}/rss/"
        f"{chart_slug}/limit={CHART_LIMIT}/genre=6014/json"
    )
    


def fetch_json(url: str) -> dict[str, Any]:
    """
    Download and validate a JSON response.

    Raises:
        requests.RequestException:
            If the network request fails or returns an HTTP error.
        ValueError:
            If the response cannot be decoded as JSON.
    """
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": (
                "GameScoutAI/1.0 "
                "(educational App Store market analysis project)"
            )
        },
    )

    response.raise_for_status()

    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Apple did not return valid JSON for URL: {url}"
        ) from error


def parse_chart(
    payload: dict[str, Any],
    chart_name: str,
    source_url: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """
    Convert Apple's legacy RSS response into flat game records.
    """
    entries = payload.get("feed", {}).get("entry", [])

    if isinstance(entries, dict):
        entries = [entries]

    if not isinstance(entries, list):
        raise ValueError(
            f"Unexpected Apple response format for chart '{chart_name}'."
        )

    records: list[dict[str, Any]] = []

    for position, game in enumerate(entries, start=1):
        app_id_attributes = game.get("id", {}).get("attributes", {})
        category_attributes = game.get("category", {}).get("attributes", {})

        images = game.get("im:image", [])
        artwork_url = None

        if images:
            artwork_url = images[-1].get("label")

        record = {
            "app_id": app_id_attributes.get("im:id"),
            "title": game.get("im:name", {}).get("label"),
            "developer": game.get("im:artist", {}).get("label"),
            "release_date": (
                game.get("im:releaseDate", {})
                .get("attributes", {})
                .get("label")
            ),
            "primary_genre": category_attributes.get("label"),
            "genres": category_attributes.get("label"),
            "artwork_url": artwork_url,
            "app_store_url": game.get("id", {}).get("label"),
            "chart_type": chart_name,
            "chart_position": position,
            "country": COUNTRY.upper(),
            "snapshot_date": snapshot_date,
            "source_url": source_url,
        }

        records.append(record)

    return records


def collect_chart(
    chart_name: str,
    chart_slug: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """
    Download and parse one App Store chart.
    """
    url = build_feed_url(chart_slug)

    print(f"\nCollecting chart: {chart_name}")
    print(f"Source: {url}")

    payload = fetch_json(url)

    records = parse_chart(
        payload=payload,
        chart_name=chart_name,
        source_url=url,
        snapshot_date=snapshot_date,
    )

    print(f"Collected {len(records)} apps.")

    return records


def save_raw_data(
    games_dataframe: pd.DataFrame,
    snapshot_timestamp: str,
) -> Path:
    """
    Save the combined raw chart data to a timestamped CSV file.
    """
    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = (
        RAW_DATA_DIRECTORY
        / f"app_store_charts_{snapshot_timestamp}.csv"
    )

    games_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def display_preview(games_dataframe: pd.DataFrame) -> None:
    """
    Display a concise preview in the terminal.
    """
    preview_columns = [
        "chart_type",
        "chart_position",
        "title",
        "developer",
        "primary_genre",
    ]

    print("\nPreview of collected data:\n")
    print(
        games_dataframe[preview_columns]
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the complete chart collection pipeline.
    """
    now_utc = datetime.now(timezone.utc)

    snapshot_date = now_utc.date().isoformat()
    snapshot_timestamp = now_utc.strftime("%Y%m%d_%H%M%S")

    all_records: list[dict[str, Any]] = []

    for chart_name, chart_slug in CHARTS.items():
        try:
            chart_records = collect_chart(
                chart_name=chart_name,
                chart_slug=chart_slug,
                snapshot_date=snapshot_date,
            )
            all_records.extend(chart_records)

        except (requests.RequestException, ValueError) as error:
            print(
                f"\nWarning: chart '{chart_name}' could not be collected."
            )
            print(f"Reason: {error}")

    if not all_records:
        raise RuntimeError(
            "No App Store data was collected. "
            "Check your internet connection and the Apple feed URLs."
        )

    games_dataframe = pd.DataFrame(all_records)

    # Ensure chart positions are interpreted as numeric values.
    games_dataframe["chart_position"] = pd.to_numeric(
        games_dataframe["chart_position"],
        errors="coerce",
    )

    display_preview(games_dataframe)

    output_path = save_raw_data(
        games_dataframe=games_dataframe,
        snapshot_timestamp=snapshot_timestamp,
    )

    unique_game_count = games_dataframe["app_id"].nunique()

    print("\nCollection completed successfully.")
    print(f"Rows collected: {len(games_dataframe)}")
    print(f"Unique games: {unique_game_count}")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()