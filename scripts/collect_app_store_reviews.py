"""
Collect public US App Store reviews for all games in GameScout AI.

The script:
1. loads all unique games from the enriched dataset;
2. queries several pages of Apple's public review RSS feed;
3. retries temporary failures;
4. stops automatically when Apple returns an empty page;
5. removes duplicate reviews;
6. saves regular checkpoints;
7. produces both:
   - app_store_reviews.csv
   - review_collection_coverage.csv

Run from the project root:

    python scripts/collect_app_store_reviews.py
"""

from __future__ import annotations
import argparse

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

COUNTRY = "us"

# Apple RSS generally returns up to ~50 reviews per page.
MAX_PAGES_PER_GAME = 5

# Maximum number of attempts when one request fails.
MAX_RETRIES = 3

REQUEST_TIMEOUT_SECONDS = 20

# Delay between two successful requests.
SECONDS_BETWEEN_REQUESTS = 0.20

# Increasing delay after a failed request.
RETRY_DELAY_SECONDS = 2.0

# Save a checkpoint every N games.
CHECKPOINT_EVERY_GAMES = 20

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-reviews",
    type=int,
    default=None,
    help="Maximum number of reviews collected per game."
)

ARGS = parser.parse_args()


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_games_enriched.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_reviews.csv"
)

COVERAGE_FILE = (
    PROCESSED_DIRECTORY
    / "review_collection_coverage.csv"
)

CHECKPOINT_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_reviews_checkpoint.csv"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_games() -> pd.DataFrame:
    """
    Load the enriched App Store dataset and keep one row per application.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The enriched dataset was not found.\n"
            "Run scripts/enrich_app_store_metadata.py first."
        )

    dataframe = pd.read_csv(INPUT_FILE)

    required_columns = {
        "app_id",
        "title",
        "developer",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe = dataframe.copy()

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    dataframe["title"] = (
        dataframe["title"]
        .fillna("Unknown game")
        .astype(str)
    )

    dataframe["developer"] = (
        dataframe["developer"]
        .fillna("Unknown")
        .astype(str)
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["app_id"]
        )
        .reset_index(drop=True)
    )

    return dataframe


# =============================================================================
# APPLE RSS URL
# =============================================================================

def build_review_url(
    app_id: str,
    page_number: int,
) -> str:
    """
    Build the Apple public review RSS URL for one application and one page.
    """
    return (
        f"https://itunes.apple.com/"
        f"{COUNTRY}/rss/customerreviews/"
        f"id={app_id}/"
        f"sortBy=mostRecent/"
        f"page={page_number}/json"
    )


# =============================================================================
# HTTP REQUESTS
# =============================================================================

def fetch_review_page(
    session: requests.Session,
    app_id: str,
    page_number: int,
) -> dict[str, Any] | None:
    """
    Download one review page with automatic retries.

    Returns:
        JSON dictionary when successful.
        None if all retries fail.
    """
    url = build_review_url(
        app_id=app_id,
        page_number=page_number,
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError(
                    "Unexpected JSON format."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print(
                f"        Attempt "
                f"{attempt}/{MAX_RETRIES} failed: "
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS
                    * attempt
                )

    return None


# =============================================================================
# RSS PARSING UTILITIES
# =============================================================================

def get_label(
    dictionary: dict[str, Any],
    key: str,
) -> str | None:
    """
    Extract a value stored by Apple as:

        {"label": "value"}
    """
    value = dictionary.get(key)

    if not isinstance(value, dict):
        return None

    label = value.get("label")

    if label is None:
        return None

    return str(label)


def extract_entries(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract review entries from the Apple RSS response.

    Apple occasionally returns metadata without any actual reviews.
    """
    feed = payload.get(
        "feed",
        {},
    )

    if not isinstance(feed, dict):
        return []

    entries = feed.get(
        "entry",
        [],
    )

    if isinstance(entries, dict):
        entries = [entries]

    if not isinstance(entries, list):
        return []

    # Some RSS entries may represent app metadata rather than reviews.
    review_entries = []

    for entry in entries:

        if not isinstance(entry, dict):
            continue

        if "im:rating" not in entry:
            continue

        if "content" not in entry:
            continue

        review_entries.append(
            entry
        )

    return review_entries


# =============================================================================
# REVIEW PARSING
# =============================================================================

def parse_review_entries(
    entries: list[dict[str, Any]],
    app_id: str,
    game_title: str,
    developer: str,
) -> list[dict[str, Any]]:
    """
    Convert nested Apple RSS reviews into flat records.
    """
    records: list[dict[str, Any]] = []

    for review in entries:

        rating_text = get_label(
            review,
            "im:rating",
        )

        review_text = get_label(
            review,
            "content",
        )

        review_title = get_label(
            review,
            "title",
        )

        review_id = get_label(
            review,
            "id",
        )

        review_date = get_label(
            review,
            "updated",
        )

        review_version = get_label(
            review,
            "im:version",
        )

        # Author uses another nested structure.
        author = None

        author_object = review.get(
            "author",
            {},
        )

        if isinstance(
            author_object,
            dict,
        ):
            author = get_label(
                author_object,
                "name",
            )

        if (
            rating_text is None
            or review_text is None
        ):
            continue

        try:
            rating = int(
                float(rating_text)
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if rating < 1 or rating > 5:
            continue

        records.append(
            {
                "app_id": app_id,
                "game_title": game_title,
                "developer": developer,
                "review_id": review_id,
                "reviewer": author,
                "review_title": review_title,
                "review_text": review_text,
                "review_rating": rating,
                "review_date": review_date,
                "review_version": review_version,
                "country": COUNTRY.upper(),
            }
        )

    return records


# =============================================================================
# SINGLE GAME COLLECTION
# =============================================================================

def collect_reviews_for_game(
    session: requests.Session,
    app_id: str,
    game_title: str,
    developer: str,
) -> tuple[
    list[dict[str, Any]],
    int,
    bool,
]:
    """
    Collect up to MAX_PAGES_PER_GAME review pages for one game.

    Returns:
        reviews
        number_of_pages_collected
        request_failed
    """
    game_reviews: list[
        dict[str, Any]
    ] = []

    pages_collected = 0
    request_failed = False

    for page_number in range(
        1,
        MAX_PAGES_PER_GAME + 1,
    ):

        payload = fetch_review_page(
            session=session,
            app_id=app_id,
            page_number=page_number,
        )

        if payload is None:
            request_failed = True

            print(
                f"        Stopping after failed "
                f"page {page_number}"
            )

            break

        entries = extract_entries(
            payload
        )

        # No reviews means there are no more pages available.
        if not entries:

            if page_number == 1:
                print(
                    "        No public reviews returned"
                )

            else:
                print(
                    f"        Page {page_number}: "
                    f"no more reviews"
                )

            break

        page_reviews = parse_review_entries(
            entries=entries,
            app_id=app_id,
            game_title=game_title,
            developer=developer,
        )

        game_reviews.extend(
            page_reviews
        )

        if (ARGS.max_reviews is not None and len(game_reviews) >= ARGS.max_reviews):
            game_reviews = game_reviews[:ARGS.max_reviews]
            break

        pages_collected += 1

        print(
            f"        Page {page_number}: "
            f"{len(page_reviews)} reviews"
        )

        # If Apple returns fewer than a full page,
        # chances are that we reached the end.
        if len(page_reviews) < 40:
            break

        time.sleep(
            SECONDS_BETWEEN_REQUESTS
        )

    return (
        game_reviews,
        pages_collected,
        request_failed,
    )


# =============================================================================
# CLEANING
# =============================================================================

def clean_reviews(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and deduplicate collected reviews.
    """
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.copy()

    dataframe[
        "review_rating"
    ] = pd.to_numeric(
        dataframe[
            "review_rating"
        ],
        errors="coerce",
    )

    dataframe[
        "review_text"
    ] = (
        dataframe[
            "review_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe[
        "review_date"
    ] = pd.to_datetime(
        dataframe[
            "review_date"
        ],
        errors="coerce",
        utc=True,
    )

    dataframe = dataframe[
        dataframe[
            "review_rating"
        ].between(
            1,
            5,
        )
    ]

    dataframe = dataframe[
        dataframe[
            "review_text"
        ].str.len() >= 5
    ]

    # Review ID is preferable, but a fallback is required because Apple can
    # occasionally return a missing ID.
    dataframe[
        "deduplication_key"
    ] = (
        dataframe[
            "app_id"
        ].astype(str)
        + "||"
        + dataframe[
            "review_id"
        ].fillna("").astype(str)
        + "||"
        + dataframe[
            "review_text"
        ].astype(str)
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "deduplication_key"
            ]
        )
        .drop(
            columns=[
                "deduplication_key"
            ]
        )
        .reset_index(drop=True)
    )

    return dataframe


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    reviews: list[dict[str, Any]],
) -> None:
    """
    Save intermediate results in case the collection is interrupted.
    """
    if not reviews:
        return

    checkpoint_dataframe = pd.DataFrame(
        reviews
    )

    checkpoint_dataframe = clean_reviews(
        checkpoint_dataframe
    )

    checkpoint_dataframe.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8",
    )


# =============================================================================
# FULL COLLECTION
# =============================================================================

def collect_all_reviews(
    games: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Collect reviews for every game and build a coverage report.
    """
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "GameScoutAI/1.0 "
                "(educational App Store market analysis)"
            )
        }
    )

    all_reviews: list[
        dict[str, Any]
    ] = []

    coverage_records: list[
        dict[str, Any]
    ] = []

    total_games = len(
        games
    )

    for index, (_, game) in enumerate(
        games.iterrows(),
        start=1,
    ):

        app_id = str(
            game["app_id"]
        )

        title = str(
            game["title"]
        )

        developer = str(
            game["developer"]
        )

        print(
            f"\n[{index}/{total_games}] "
            f"{title}"
        )

        (
            game_reviews,
            pages_collected,
            request_failed,
        ) = collect_reviews_for_game(
            session=session,
            app_id=app_id,
            game_title=title,
            developer=developer,
        )

        all_reviews.extend(
            game_reviews
        )

        coverage_records.append(
            {
                "app_id": app_id,
                "game_title": title,
                "developer": developer,
                "reviews_collected": len(
                    game_reviews
                ),
                "pages_collected": (
                    pages_collected
                ),
                "request_failed": (
                    request_failed
                ),
            }
        )

        print(
            f"    TOTAL: "
            f"{len(game_reviews)} reviews"
        )

        # Regular checkpoint to protect against interruption.
        if (
            index % CHECKPOINT_EVERY_GAMES
            == 0
        ):
            save_checkpoint(
                all_reviews
            )

            print(
                f"\n    ✓ Checkpoint saved "
                f"after {index} games"
            )

        time.sleep(
            SECONDS_BETWEEN_REQUESTS
        )

    reviews_dataframe = pd.DataFrame(
        all_reviews
    )

    coverage_dataframe = pd.DataFrame(
        coverage_records
    )

    return (
        reviews_dataframe,
        coverage_dataframe,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def display_summary(
    reviews: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    """
    Print the final collection statistics.
    """
    games_requested = len(
        coverage
    )

    games_with_reviews = int(
        (
            coverage[
                "reviews_collected"
            ] > 0
        ).sum()
    )

    games_without_reviews = (
        games_requested
        - games_with_reviews
    )

    total_reviews = len(
        reviews
    )

    print("\n")
    print("=" * 60)
    print("REVIEW COLLECTION SUMMARY")
    print("=" * 60)

    print(
        f"Games requested: "
        f"{games_requested}"
    )

    print(
        f"Games with reviews: "
        f"{games_with_reviews}"
    )

    print(
        f"Games without reviews: "
        f"{games_without_reviews}"
    )

    print(
        f"Total reviews collected: "
        f"{total_reviews}"
    )

    if not reviews.empty:

        average_rating = reviews[
            "review_rating"
        ].mean()

        negative_reviews = int(
            (
                reviews[
                    "review_rating"
                ] <= 2
            ).sum()
        )

        neutral_reviews = int(
            (
                reviews[
                    "review_rating"
                ] == 3
            ).sum()
        )

        positive_reviews = int(
            (
                reviews[
                    "review_rating"
                ] >= 4
            ).sum()
        )

        print(
            f"Average review rating: "
            f"{average_rating:.2f}"
        )

        print(
            f"Negative reviews (1-2 stars): "
            f"{negative_reviews}"
        )

        print(
            f"Neutral reviews (3 stars): "
            f"{neutral_reviews}"
        )

        print(
            f"Positive reviews (4-5 stars): "
            f"{positive_reviews}"
        )

        print(
            f"Unique apps represented: "
            f"{reviews['app_id'].nunique()}"
        )

    failed_requests = int(
        coverage[
            "request_failed"
        ].sum()
    )

    print(
        f"Games with request failures: "
        f"{failed_requests}"
    )

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Run the complete review collection pipeline.
    """
    print(
        "Loading enriched App Store games..."
    )

    games = load_games()

    print(
        f"Unique games to query: "
        f"{len(games)}"
    )

    print(
        f"Maximum pages per game: "
        f"{MAX_PAGES_PER_GAME}"
    )

    theoretical_maximum = (
        len(games)
        * MAX_PAGES_PER_GAME
        * 50
    )

    print(
        f"Theoretical maximum reviews: "
        f"{theoretical_maximum:,}"
    )

    reviews, coverage = (
        collect_all_reviews(
            games
        )
    )

    if reviews.empty:
        raise RuntimeError(
            "No reviews were collected."
        )

    print(
        "\nCleaning and deduplicating reviews..."
    )

    reviews = clean_reviews(
        reviews
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
        encoding="utf-8",
    )

    display_summary(
        reviews=reviews,
        coverage=coverage,
    )

    print(
        f"\nReviews saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nCoverage report saved to:\n"
        f"{COVERAGE_FILE}"
    )

    # The temporary checkpoint is no longer necessary after success.
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


if __name__ == "__main__":
    main()