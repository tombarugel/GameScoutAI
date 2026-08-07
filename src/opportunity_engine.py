"""
Opportunity-scoring engine for GameScout AI.

This version transforms TF-IDF game clusters into interpretable market signals.

Important:
    The Opportunity Score is a heuristic based on the current US App Store
    snapshot. It is not a revenue forecast, download forecast, or proof of
    market growth.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_WEIGHTS = {
    "market_strength": 25,
    "player_validation": 20,
    "competition_gap": 15,
    "developer_diversity": 15,
    "semantic_coherence": 15,
    "confidence": 10,
}


GENERIC_CLUSTER_KEYWORDS = {
    "action",
    "adventure",
    "app",
    "best",
    "casual",
    "experience",
    "features",
    "friends",
    "game",
    "games",
    "great",
    "items",
    "make",
    "mode",
    "new",
    "play",
    "players",
    "playing",
    "story",
    "use",
    "world",
}


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def min_max_normalize(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Normalize a numeric series to the [0, 1] interval.
    """
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):
        normalized = pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    else:
        normalized = (
            numeric_series - minimum
        ) / (
            maximum - minimum
        )

    normalized = normalized.fillna(0.5)

    if not higher_is_better:
        normalized = 1 - normalized

    return normalized.clip(0, 1)


def split_pipe_separated_text(
    value: object,
) -> list[str]:
    """
    Turn a pipe-separated CSV field into a list.
    """
    if value is None or pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


# =============================================================================
# CLUSTER NAMING
# =============================================================================

def is_useful_cluster_keyword(
    keyword: str,
) -> bool:
    """
    Decide whether a keyword is informative enough for a cluster name.
    """
    normalized = keyword.lower().strip()

    if not normalized:
        return False

    if normalized in GENERIC_CLUSTER_KEYWORDS:
        return False

    if not re.search(r"[a-zA-Z]", normalized):
        return False

    return True


def generate_rule_based_cluster_name(
    keywords: object,
    representative_games: object,
    top_developer: str,
    cluster_id: int,
) -> str:
    """
    Build a readable cluster name using simple deterministic rules.

    Specific franchise rules are applied only when the evidence is obvious.
    """
    keyword_list = [
        keyword.lower()
        for keyword in split_pipe_separated_text(
            keywords
        )
    ]

    games_text = str(
        representative_games
    ).lower()

    developer_text = str(
        top_developer
    ).lower()

    combined_text = (
        " ".join(keyword_list)
        + " "
        + games_text
        + " "
        + developer_text
    )

    # -------------------------------------------------------------------------
    # Strong franchise / domain patterns
    # -------------------------------------------------------------------------

    if (
        "freddy" in combined_text
        or "fazbear" in combined_text
        or "five nights" in combined_text
    ):
        return "Five Nights at Freddy's Franchise"

    if (
        "flipline" in developer_text
        or "papa's" in combined_text
        or "papas" in combined_text
    ):
        return "Papa's Cooking & Time Management"

    if (
        "casino" in combined_text
        or "cash" in combined_text
        or "slots" in combined_text
    ):
        return "Casino & Real-Money Games"

    if (
        "puzzle" in combined_text
        and (
            "sort" in combined_text
            or "block" in combined_text
            or "brain" in combined_text
        )
    ):
        return "Casual Puzzle & Sorting"

    if (
        "word" in combined_text
        or "crossword" in combined_text
        or "clue" in combined_text
    ):
        return "Word & Brain Games"

    if (
        "rpg" in combined_text
        or "boss" in combined_text
        or "battle" in combined_text
    ):
        return "RPG & Combat Games"

    if (
        "strategy" in combined_text
        and (
            "simulation" in combined_text
            or "management" in combined_text
            or "build" in combined_text
        )
    ):
        return "Strategy & Simulation"

    # -------------------------------------------------------------------------
    # Generic fallback based on top useful keywords
    # -------------------------------------------------------------------------

    selected_keywords: list[str] = []

    for keyword in keyword_list:
        if not is_useful_cluster_keyword(
            keyword
        ):
            continue

        already_represented = any(
            keyword in selected.lower()
            or selected.lower() in keyword
            for selected in selected_keywords
        )

        if already_represented:
            continue

        selected_keywords.append(
            keyword.title()
        )

        if len(selected_keywords) >= 3:
            break

    if not selected_keywords:
        return f"Game Segment {cluster_id}"

    return " / ".join(
        selected_keywords
    )


# =============================================================================
# DEVELOPER CONCENTRATION
# =============================================================================

def calculate_developer_concentration(
    clustered_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the share belonging to the largest developer in each cluster.
    """
    records: list[dict[str, object]] = []

    for cluster_id, cluster in clustered_games.groupby(
        "cluster_id"
    ):

        distribution = (
            cluster["developer"]
            .fillna("Unknown")
            .value_counts(normalize=True)
        )

        if distribution.empty:
            top_developer = "Unknown"
            top_developer_share = 1.0

        else:
            top_developer = str(
                distribution.index[0]
            )

            top_developer_share = float(
                distribution.iloc[0]
            )

        records.append(
            {
                "cluster_id": int(cluster_id),
                "top_developer": top_developer,
                "top_developer_share": (
                    top_developer_share
                ),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# SEMANTIC COHERENCE
# =============================================================================

def calculate_cluster_coherence(
    clustered_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate semantic coherence inside each cluster.

    A TF-IDF representation is built from the cleaned text already produced
    during clustering. For every cluster, each game's cosine similarity to the
    cluster centroid is calculated.

    Higher values mean games inside the cluster use more similar semantic
    vocabulary.

    This score is used as an interpretability / reliability signal rather than
    as a clustering-performance benchmark.
    """
    if "cleaned_text" not in clustered_games.columns:
        raise ValueError(
            "The clustered-games dataset must contain "
            "'cleaned_text'. Re-run build_game_clusters.py."
        )

    texts = (
        clustered_games["cleaned_text"]
        .fillna("")
        .astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        max_features=5000,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        texts
    )

    records: list[dict[str, float]] = []

    for cluster_id in sorted(
        clustered_games["cluster_id"].unique()
    ):

        cluster_indices = np.where(
            clustered_games[
                "cluster_id"
            ].to_numpy()
            == cluster_id
        )[0]

        cluster_matrix = tfidf_matrix[
            cluster_indices
        ]

        if len(cluster_indices) <= 1:
            coherence = 0.0

        else:
            centroid = np.asarray(
                cluster_matrix.mean(axis=0)
            )

            similarities = cosine_similarity(
                cluster_matrix,
                centroid,
            ).ravel()

            coherence = float(
                similarities.mean()
            )

        records.append(
            {
                "cluster_id": int(cluster_id),
                "semantic_coherence": coherence,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# SCORE COMPONENTS
# =============================================================================

def calculate_market_strength(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Higher score for clusters appearing higher in current rankings.
    """
    return min_max_normalize(
        dataframe["average_chart_position"],
        higher_is_better=False,
    )


def calculate_player_validation(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Combine ratings quality and rating volume.
    """
    rating_quality = min_max_normalize(
        dataframe["average_rating"],
        higher_is_better=True,
    )

    rating_volume_raw = np.log1p(
        pd.to_numeric(
            dataframe["median_rating_count"],
            errors="coerce",
        ).fillna(0)
    )

    rating_volume = min_max_normalize(
        rating_volume_raw,
        higher_is_better=True,
    )

    return (
        0.65 * rating_quality
        + 0.35 * rating_volume
    )


def calculate_competition_gap(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Estimate underrepresentation in current charts.

    Important:
        A franchise-dominated cluster receives a major penalty because a small
        number of games does not necessarily indicate an underserved market.
    """
    game_scarcity = min_max_normalize(
        dataframe["game_count"],
        higher_is_better=False,
    )

    developer_scarcity = min_max_normalize(
        dataframe["developer_count"],
        higher_is_better=False,
    )

    raw_gap = (
        0.55 * game_scarcity
        + 0.45 * developer_scarcity
    )

    # Franchise penalty:
    # 50% concentration => substantial penalty.
    # 100% concentration => only 20% of the original gap remains.
    concentration_penalty = (
        1
        - 0.80
        * dataframe[
            "top_developer_share"
        ].clip(0, 1)
    )

    return (
        raw_gap
        * concentration_penalty
    ).clip(0, 1)


def calculate_developer_diversity(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Reward cross-developer segments.
    """
    return (
        1
        - dataframe["top_developer_share"]
    ).clip(0, 1)


def calculate_confidence(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Estimate statistical confidence from cluster size.

    15 games or more gives maximum size confidence.
    """
    game_count = pd.to_numeric(
        dataframe["game_count"],
        errors="coerce",
    ).fillna(0)

    return (
        game_count / 15
    ).clip(0, 1)


# =============================================================================
# LABELS
# =============================================================================

def classify_confidence_level(
    game_count: int,
    developer_count: int,
    semantic_coherence: float,
) -> str:
    """
    Convert numeric reliability signals into a readable label.
    """
    if (
        game_count >= 15
        and developer_count >= 8
        and semantic_coherence >= 0.25
    ):
        return "High"

    if (
        game_count >= 7
        and developer_count >= 4
        and semantic_coherence >= 0.15
    ):
        return "Medium"

    return "Low"


def classify_segment_type(
    row: pd.Series,
) -> str:
    """
    Classify the nature of each cluster.
    """
    concentration = float(
        row["top_developer_share"]
    )

    coherence = float(
        row["semantic_coherence"]
    )

    game_count = int(
        row["game_count"]
    )

    if concentration >= 0.50:
        return "Franchise cluster"

    if coherence < 0.15:
        return "Mixed / low-confidence cluster"

    if (
        game_count >= 20
        and row["competition_gap_score"] < 7
    ):
        return "Established competitive segment"

    return "Opportunity candidate"


def build_warning(
    row: pd.Series,
) -> str:
    """
    Produce a concise warning when interpretation requires caution.
    """
    warnings: list[str] = []

    if row["top_developer_share"] >= 0.50:
        warnings.append(
            "High developer/franchise concentration"
        )

    if row["semantic_coherence"] < 0.15:
        warnings.append(
            "Low semantic coherence"
        )

    if row["game_count"] < 5:
        warnings.append(
            "Very small cluster"
        )

    return "; ".join(
        warnings
    )


def build_signal_summary(
    row: pd.Series,
) -> str:
    """
    Explain why a cluster received its score.
    """
    signals: list[str] = []

    if row["market_strength_score"] >= 17:
        signals.append(
            "strong current chart visibility"
        )

    if row["player_validation_score"] >= 14:
        signals.append(
            "strong player validation"
        )

    if row["competition_gap_score"] >= 9:
        signals.append(
            "relatively limited representation"
        )

    if row["developer_diversity_score"] >= 10:
        signals.append(
            "diverse developer landscape"
        )

    if row["semantic_coherence_score"] >= 10:
        signals.append(
            "coherent game segment"
        )

    if not signals:
        return (
            "Mixed market signals; further validation recommended."
        )

    return (
        "; ".join(signals).capitalize()
        + "."
    )


# =============================================================================
# MAIN OPPORTUNITY ENGINE
# =============================================================================

def build_opportunity_scores(
    cluster_summary: pd.DataFrame,
    clustered_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate interpretable Opportunity Scores for all clusters.
    """
    required_summary_columns = {
        "cluster_id",
        "game_count",
        "developer_count",
        "average_rating",
        "median_rating_count",
        "average_chart_position",
        "keywords",
        "representative_games",
    }

    missing = required_summary_columns.difference(
        cluster_summary.columns
    )

    if missing:
        raise ValueError(
            "cluster_summary.csv is missing: "
            + ", ".join(sorted(missing))
        )

    dataframe = cluster_summary.copy()

    concentration = (
        calculate_developer_concentration(
            clustered_games
        )
    )

    coherence = (
        calculate_cluster_coherence(
            clustered_games
        )
    )

    dataframe = dataframe.merge(
        concentration,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    dataframe = dataframe.merge(
        coherence,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Automatic cluster names
    # -------------------------------------------------------------------------

    dataframe["cluster_name"] = dataframe.apply(
        lambda row: generate_rule_based_cluster_name(
            keywords=row["keywords"],
            representative_games=row[
                "representative_games"
            ],
            top_developer=row[
                "top_developer"
            ],
            cluster_id=int(
                row["cluster_id"]
            ),
        ),
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Raw normalized components
    # -------------------------------------------------------------------------

    market_strength = (
        calculate_market_strength(
            dataframe
        )
    )

    player_validation = (
        calculate_player_validation(
            dataframe
        )
    )

    competition_gap = (
        calculate_competition_gap(
            dataframe
        )
    )

    developer_diversity = (
        calculate_developer_diversity(
            dataframe
        )
    )

    semantic_coherence = min_max_normalize(
        dataframe[
            "semantic_coherence"
        ],
        higher_is_better=True,
    )

    confidence = calculate_confidence(
        dataframe
    )

    # -------------------------------------------------------------------------
    # Weighted components
    # -------------------------------------------------------------------------

    dataframe[
        "market_strength_score"
    ] = (
        market_strength
        * SCORE_WEIGHTS[
            "market_strength"
        ]
    )

    dataframe[
        "player_validation_score"
    ] = (
        player_validation
        * SCORE_WEIGHTS[
            "player_validation"
        ]
    )

    dataframe[
        "competition_gap_score"
    ] = (
        competition_gap
        * SCORE_WEIGHTS[
            "competition_gap"
        ]
    )

    dataframe[
        "developer_diversity_score"
    ] = (
        developer_diversity
        * SCORE_WEIGHTS[
            "developer_diversity"
        ]
    )

    dataframe[
        "semantic_coherence_score"
    ] = (
        semantic_coherence
        * SCORE_WEIGHTS[
            "semantic_coherence"
        ]
    )

    dataframe[
        "confidence_score"
    ] = (
        confidence
        * SCORE_WEIGHTS[
            "confidence"
        ]
    )

    score_columns = [
        "market_strength_score",
        "player_validation_score",
        "competition_gap_score",
        "developer_diversity_score",
        "semantic_coherence_score",
        "confidence_score",
    ]

    dataframe[
        "opportunity_score"
    ] = (
        dataframe[
            score_columns
        ]
        .sum(axis=1)
        .round(1)
    )

    for column in score_columns:
        dataframe[column] = (
            dataframe[column]
            .round(1)
        )

    dataframe[
        "semantic_coherence"
    ] = dataframe[
        "semantic_coherence"
    ].round(3)

    # -------------------------------------------------------------------------
    # Interpretability
    # -------------------------------------------------------------------------

    dataframe[
        "confidence_level"
    ] = dataframe.apply(
        lambda row: classify_confidence_level(
            game_count=int(
                row["game_count"]
            ),
            developer_count=int(
                row["developer_count"]
            ),
            semantic_coherence=float(
                row[
                    "semantic_coherence"
                ]
            ),
        ),
        axis=1,
    )

    dataframe[
        "segment_type"
    ] = dataframe.apply(
        classify_segment_type,
        axis=1,
    )

    dataframe[
        "warning"
    ] = dataframe.apply(
        build_warning,
        axis=1,
    )

    dataframe[
        "signal_summary"
    ] = dataframe.apply(
        build_signal_summary,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Product ranking
    # -------------------------------------------------------------------------

    # Franchise and mixed clusters remain visible but should not rank above
    # reliable opportunity candidates purely because they are small.
    dataframe[
        "ranking_penalty"
    ] = 0.0

    dataframe.loc[
        dataframe[
            "segment_type"
        ] == "Franchise cluster",
        "ranking_penalty",
    ] = 15.0

    dataframe.loc[
        dataframe[
            "segment_type"
        ]
        == "Mixed / low-confidence cluster",
        "ranking_penalty",
    ] = 20.0

    dataframe[
        "adjusted_opportunity_score"
    ] = (
        dataframe[
            "opportunity_score"
        ]
        - dataframe[
            "ranking_penalty"
        ]
    ).clip(
        lower=0
    ).round(1)

    dataframe = dataframe.sort_values(
        "adjusted_opportunity_score",
        ascending=False,
    ).reset_index(drop=True)

    dataframe.insert(
        0,
        "opportunity_rank",
        range(
            1,
            len(dataframe) + 1,
        ),
    )

    return dataframe