"""Data access helpers for the GameScout AI Streamlit application.

The app is intentionally backed by cached CSV snapshots committed with the project.
This makes the evaluator experience immediate and reproducible, while the scripts/
folder remains available to refresh the data pipeline when needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


FILES = {
    "enriched_games": PROCESSED_DIR / "app_store_games_enriched.csv",
    "clustered_games": PROCESSED_DIR / "app_store_games_clustered.csv",
    "cluster_summary": PROCESSED_DIR / "cluster_summary.csv",
    "opportunities": PROCESSED_DIR / "opportunity_scores.csv",
    "reviews": PROCESSED_DIR / "app_store_reviews.csv",
    "review_coverage": PROCESSED_DIR / "review_collection_coverage.csv",
    "pain_point_summary": PROCESSED_DIR / "pain_point_summary.csv",
    "cluster_pain_points": PROCESSED_DIR / "cluster_pain_points.csv",
}


def _read_csv(path: Path, required: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required project file not found: {path}")

    dataframe = pd.read_csv(path)

    if required:
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(sorted(missing))}"
            )

    return dataframe


def load_app_data() -> dict[str, pd.DataFrame]:
    """Load all application datasets used by the UI."""
    return {
        "enriched_games": _read_csv(
            FILES["enriched_games"],
            {"app_id", "title", "developer", "chart_type", "chart_position"},
        ),
        "clustered_games": _read_csv(
            FILES["clustered_games"],
            {"app_id", "title", "developer", "cluster_id"},
        ),
        "cluster_summary": _read_csv(
            FILES["cluster_summary"],
            {"cluster_id", "game_count", "keywords", "representative_games"},
        ),
        "opportunities": _read_csv(
            FILES["opportunities"],
            {
                "cluster_id",
                "cluster_name",
                "adjusted_opportunity_score",
                "segment_type",
                "representative_games",
            },
        ),
        "reviews": _read_csv(
            FILES["reviews"],
            {"app_id", "game_title", "review_text", "review_rating"},
        ),
        "review_coverage": _read_csv(
            FILES["review_coverage"],
            {"app_id", "game_title", "reviews_collected"},
        ),
        "pain_point_summary": _read_csv(
            FILES["pain_point_summary"],
            {"pain_point_id", "keywords", "representative_reviews"},
        ),
        "cluster_pain_points": _read_csv(
            FILES["cluster_pain_points"],
            {"cluster_id", "pain_point_id", "pain_point_share"},
        ),
    }


def split_pipe(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def get_cluster_games(data: dict[str, pd.DataFrame], cluster_id: int) -> pd.DataFrame:
    games = data["clustered_games"]
    return games[games["cluster_id"] == cluster_id].copy()


def get_cluster_pain_points(
    data: dict[str, pd.DataFrame],
    cluster_id: int,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Return review pain points available for one game cluster.

    The public Apple review feed has incomplete coverage, therefore evidence is only
    returned when the pipeline marked the cluster as sufficiently covered.
    """
    links = data["cluster_pain_points"]
    rows = links[links["cluster_id"] == cluster_id].copy()

    if rows.empty:
        return []

    if "sufficient_review_coverage" in rows.columns:
        usable = rows["sufficient_review_coverage"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        rows = rows[usable]

    if rows.empty:
        return []

    rows = rows.sort_values("pain_point_share", ascending=False).head(max_items)
    summary = data["pain_point_summary"]
    rows = rows.merge(summary, on="pain_point_id", how="left")

    results: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        examples = [
            item.strip()
            for item in str(row.get("representative_reviews", "")).split(" || ")
            if item.strip()
        ]
        results.append(
            {
                "pain_point_id": int(row["pain_point_id"]),
                "share": float(row["pain_point_share"]),
                "keywords": str(row.get("keywords", "")),
                "example_reviews": examples[:2],
                "cluster_negative_reviews": int(row.get("cluster_negative_reviews", 0)),
            }
        )

    return results


def latest_snapshot_name() -> str:
    snapshots = list(RAW_DIR.glob("*.csv"))
    if not snapshots:
        return "No raw snapshot found"
    latest = max(snapshots, key=lambda path: path.stat().st_mtime)
    return latest.name
