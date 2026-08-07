"""
Basic market indicators calculated from the App Store chart snapshot.

This first analytical layer intentionally relies only on information observed
in the rankings. More advanced metadata, reviews and mechanics will be added
later.
"""

from __future__ import annotations

import pandas as pd


def calculate_market_metrics(dataframe: pd.DataFrame) -> dict[str, int]:
    """
    Calculate headline metrics displayed in the interface.
    """
    free_games = dataframe[
        dataframe["chart_type"] == "top_free"
    ]

    paid_games = dataframe[
        dataframe["chart_type"] == "top_paid"
    ]

    return {
        "total_rows": len(dataframe),
        "unique_games": dataframe["app_id"].nunique(),
        "free_games": free_games["app_id"].nunique(),
        "paid_games": paid_games["app_id"].nunique(),
        "developers": dataframe["developer"].nunique(),
    }


def get_top_games(
    dataframe: pd.DataFrame,
    chart_type: str,
    number_of_games: int = 10,
) -> pd.DataFrame:
    """
    Return the highest-ranked games from a selected chart.
    """
    selected_chart = dataframe[
        dataframe["chart_type"] == chart_type
    ].copy()

    return (
        selected_chart
        .sort_values("chart_position")
        .head(number_of_games)
    )


def get_developer_presence(
    dataframe: pd.DataFrame,
    number_of_developers: int = 10,
) -> pd.DataFrame:
    """
    Count how many ranked games belong to each developer.
    """
    developer_counts = (
        dataframe
        .dropna(subset=["developer"])
        .groupby("developer")
        .agg(
            ranked_games=("app_id", "nunique"),
            chart_entries=("app_id", "size"),
        )
        .reset_index()
        .sort_values(
            by=["ranked_games", "chart_entries"],
            ascending=False,
        )
        .head(number_of_developers)
    )

    return developer_counts