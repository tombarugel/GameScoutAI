"""
Build Opportunity Scores from the current TF-IDF clustering results.

The output is fully local and deterministic. No language-model API is required.

Run from the project root:

    python scripts/build_opportunity_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.opportunity_engine import (
    build_opportunity_scores,
)


PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CLUSTER_SUMMARY_FILE = (
    PROCESSED_DIRECTORY
    / "cluster_summary.csv"
)

CLUSTERED_GAMES_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_games_clustered.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIRECTORY
    / "opportunity_scores.csv"
)


def load_required_csv(
    path: Path,
    file_description: str,
) -> pd.DataFrame:
    """
    Load a required CSV file with a clear error message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{file_description} was not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def display_opportunity_ranking(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display the final ranked market segments.
    """
    print("\nOpportunity ranking")
    print("===================\n")

    for _, opportunity in dataframe.iterrows():

        print(
            f"#{int(opportunity['opportunity_rank'])} "
            f"— {opportunity['cluster_name']}"
        )

        print(
            f"  Adjusted score: "
            f"{opportunity['adjusted_opportunity_score']}/100"
        )

        print(
            f"  Raw score: "
            f"{opportunity['opportunity_score']}/100"
        )

        print(
            f"  Segment type: "
            f"{opportunity['segment_type']}"
        )

        print(
            "  Market strength: "
            f"{opportunity['market_strength_score']:.1f}/25"
        )

        print(
            "  Player validation: "
            f"{opportunity['player_validation_score']:.1f}/20"
        )

        print(
            "  Competition gap: "
            f"{opportunity['competition_gap_score']:.1f}/15"
        )

        print(
            "  Developer diversity: "
            f"{opportunity['developer_diversity_score']:.1f}/15"
        )

        print(
            "  Semantic coherence: "
            f"{opportunity['semantic_coherence_score']:.1f}/15 "
            f"(raw={opportunity['semantic_coherence']})"
        )

        print(
            "  Confidence: "
            f"{opportunity['confidence_score']:.1f}/10 "
            f"({opportunity['confidence_level']})"
        )

        print(
            f"  Signal: "
            f"{opportunity['signal_summary']}"
        )

        warning = opportunity[
            "warning"
        ]

        if isinstance(
            warning,
            str,
        ) and warning:
            print(
                f"  Warning: {warning}"
            )

        print(
            "  Representative games: "
            f"{opportunity['representative_games']}"
        )

        print("-" * 80)


def main() -> None:
    """
    Run the full opportunity-scoring pipeline.
    """
    print("Loading TF-IDF clustering results...")

    cluster_summary = load_required_csv(
        path=CLUSTER_SUMMARY_FILE,
        file_description="Cluster summary",
    )

    clustered_games = load_required_csv(
        path=CLUSTERED_GAMES_FILE,
        file_description="Clustered-games dataset",
    )

    print(
        f"Clusters loaded: {len(cluster_summary)}"
    )

    opportunity_scores = build_opportunity_scores(
        cluster_summary=cluster_summary,
        clustered_games=clustered_games,
    )

    opportunity_scores.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    display_opportunity_ranking(
        opportunity_scores
    )

    print("\nOpportunity scoring completed successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()