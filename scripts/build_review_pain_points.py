"""
Build recurring player pain points from negative App Store reviews.

Pipeline:
1. Load collected App Store reviews.
2. Keep only 1-star and 2-star reviews.
3. Clean the review text.
4. Convert reviews into TF-IDF features.
5. Evaluate several KMeans configurations.
6. Select a compact clustering solution.
7. Extract keywords and representative reviews for each pain point.
8. Link pain points back to the existing game clusters.
9. Save interpretable CSV outputs.

Run from the project root:

    python scripts/build_review_pain_points.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

MIN_CLUSTERS = 3
MAX_CLUSTERS = 10

NUMBER_OF_KEYWORDS = 10
NUMBER_OF_EXAMPLE_REVIEWS = 4

# We do not make cluster-level conclusions when too few negative reviews exist.
MIN_NEGATIVE_REVIEWS_FOR_CLUSTER_ANALYSIS = 8


PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REVIEWS_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_reviews.csv"
)

CLUSTERED_GAMES_FILE = (
    PROCESSED_DIRECTORY
    / "app_store_games_clustered.csv"
)

PAIN_POINT_REVIEWS_FILE = (
    PROCESSED_DIRECTORY
    / "review_pain_points.csv"
)

PAIN_POINT_SUMMARY_FILE = (
    PROCESSED_DIRECTORY
    / "pain_point_summary.csv"
)

CLUSTER_PAIN_POINTS_FILE = (
    PROCESSED_DIRECTORY
    / "cluster_pain_points.csv"
)

PAIN_POINT_EVALUATION_FILE = (
    PROCESSED_DIRECTORY
    / "pain_point_evaluation.csv"
)


# =============================================================================
# DOMAIN STOPWORDS
# =============================================================================

REVIEW_STOPWORDS = {
    "app",
    "apps",
    "game",
    "games",
    "gaming",
    "play",
    "played",
    "playing",
    "player",
    "players",
    "iphone",
    "ipad",
    "apple",
    "phone",
    "really",
    "just",
    "like",
    "love",
    "good",
    "great",
    "bad",
    "thing",
    "things",
    "time",
    "times",
    "want",
    "wanted",
    "make",
    "makes",
    "making",
    "got",
    "get",
    "gets",
    "getting",
}


URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+",
    flags=re.IGNORECASE,
)

NON_LETTER_PATTERN = re.compile(
    r"[^a-zA-Z\s'-]"
)

MULTIPLE_SPACES_PATTERN = re.compile(
    r"\s+"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reviews() -> pd.DataFrame:
    """
    Load and validate collected App Store reviews.
    """
    if not REVIEWS_FILE.exists():
        raise FileNotFoundError(
            "app_store_reviews.csv was not found."
        )

    dataframe = pd.read_csv(
        REVIEWS_FILE
    )

    required_columns = {
        "app_id",
        "game_title",
        "review_text",
        "review_rating",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Review dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    dataframe = dataframe.copy()

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    dataframe["review_rating"] = pd.to_numeric(
        dataframe["review_rating"],
        errors="coerce",
    )

    dataframe["review_text"] = (
        dataframe["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return dataframe


def load_game_clusters() -> pd.DataFrame:
    """
    Load the existing TF-IDF game-clustering output.
    """
    if not CLUSTERED_GAMES_FILE.exists():
        raise FileNotFoundError(
            "app_store_games_clustered.csv was not found."
        )

    dataframe = pd.read_csv(
        CLUSTERED_GAMES_FILE
    )

    required_columns = {
        "app_id",
        "cluster_id",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Clustered-games dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    dataframe = dataframe.copy()

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    dataframe = dataframe.drop_duplicates(
        subset=["app_id"]
    )

    return dataframe


# =============================================================================
# REVIEW CLEANING
# =============================================================================

def clean_review_text(
    text: object,
) -> str:
    """
    Clean one user review for NLP analysis.
    """
    if text is None:
        return ""

    text = str(text).lower()

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = NON_LETTER_PATTERN.sub(
        " ",
        text,
    )

    text = MULTIPLE_SPACES_PATTERN.sub(
        " ",
        text,
    ).strip()

    cleaned_words: list[str] = []

    for word in text.split():

        word = word.strip(
            "'-"
        )

        if len(word) < 3:
            continue

        if word in REVIEW_STOPWORDS:
            continue

        cleaned_words.append(
            word
        )

    return " ".join(
        cleaned_words
    )


def prepare_negative_reviews(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep 1-star and 2-star reviews and prepare cleaned NLP text.
    """
    negative_reviews = reviews[
        reviews["review_rating"] <= 2
    ].copy()

    negative_reviews[
        "cleaned_review"
    ] = negative_reviews[
        "review_text"
    ].apply(
        clean_review_text
    )

    # Tiny reviews such as "bad" or "trash" have almost no analytical value.
    negative_reviews = negative_reviews[
        negative_reviews[
            "cleaned_review"
        ].str.len() >= 15
    ].copy()

    negative_reviews = (
        negative_reviews
        .drop_duplicates(
            subset=[
                "app_id",
                "review_text",
            ]
        )
        .reset_index(drop=True)
    )

    if len(negative_reviews) < 20:
        raise ValueError(
            "Too few usable negative reviews for clustering."
        )

    return negative_reviews


# =============================================================================
# TF-IDF
# =============================================================================

def create_tfidf_features(
    reviews: pd.DataFrame,
) -> tuple[TfidfVectorizer, object]:
    """
    Transform negative reviews into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),

        # A complaint must appear in at least two reviews.
        min_df=2,

        # Extremely widespread words are not discriminating.
        max_df=0.80,

        max_features=5000,

        sublinear_tf=True,

        token_pattern=(
            r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b"
        ),
    )

    matrix = vectorizer.fit_transform(
        reviews["cleaned_review"]
    )

    if matrix.shape[1] < 10:
        raise ValueError(
            "TF-IDF generated too few review features."
        )

    return (
        vectorizer,
        matrix,
    )


# =============================================================================
# CLUSTER EVALUATION
# =============================================================================

def evaluate_cluster_counts(
    tfidf_matrix: object,
) -> pd.DataFrame:
    """
    Evaluate several KMeans pain-point cluster counts.
    """
    maximum_clusters = min(
        MAX_CLUSTERS,
        tfidf_matrix.shape[0] - 1,
    )

    records: list[
        dict[str, float]
    ] = []

    print(
        "\nEvaluating pain-point cluster counts:"
    )

    for number_of_clusters in range(
        MIN_CLUSTERS,
        maximum_clusters + 1,
    ):

        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=RANDOM_STATE,
            n_init=30,
        )

        labels = model.fit_predict(
            tfidf_matrix
        )

        score = silhouette_score(
            tfidf_matrix,
            labels,
            metric="cosine",
        )

        cluster_sizes = np.bincount(
            labels
        )

        records.append(
            {
                "number_of_clusters": (
                    number_of_clusters
                ),
                "silhouette_score": score,
                "smallest_cluster_size": int(
                    cluster_sizes.min()
                ),
                "largest_cluster_size": int(
                    cluster_sizes.max()
                ),
            }
        )

        print(
            f"  k={number_of_clusters:2d}"
            f" | silhouette={score:.4f}"
            f" | min={cluster_sizes.min():3d}"
            f" | max={cluster_sizes.max():3d}"
        )

    return pd.DataFrame(
        records
    )


def select_cluster_count(
    evaluation: pd.DataFrame,
) -> int:
    """
    Select a compact and interpretable pain-point solution.

    We prefer the smallest k reaching at least 95% of the best silhouette
    score, provided every cluster contains at least 10 reviews.
    """
    valid_models = evaluation[
        evaluation[
            "smallest_cluster_size"
        ] >= 10
    ].copy()

    if valid_models.empty:
        valid_models = evaluation.copy()

    best_score = valid_models[
        "silhouette_score"
    ].max()

    threshold = (
        best_score * 0.95
    )

    acceptable = (
        valid_models[
            valid_models[
                "silhouette_score"
            ] >= threshold
        ]
        .sort_values(
            "number_of_clusters"
        )
    )

    selected_k = int(
        acceptable.iloc[0][
            "number_of_clusters"
        ]
    )

    selected_score = float(
        acceptable.iloc[0][
            "silhouette_score"
        ]
    )

    print(
        f"\nBest silhouette score: "
        f"{best_score:.4f}"
    )

    print(
        f"Selected k: "
        f"{selected_k}"
    )

    print(
        f"Selected score: "
        f"{selected_score:.4f}"
    )

    return selected_k


# =============================================================================
# FINAL MODEL
# =============================================================================

def train_pain_point_model(
    tfidf_matrix: object,
    number_of_clusters: int,
) -> tuple[KMeans, np.ndarray]:
    """
    Train the final pain-point KMeans model.
    """
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    labels = model.fit_predict(
        tfidf_matrix
    )

    return (
        model,
        labels,
    )


# =============================================================================
# PAIN-POINT INTERPRETATION
# =============================================================================

def extract_keywords(
    tfidf_matrix: object,
    labels: np.ndarray,
    feature_names: np.ndarray,
    pain_point_id: int,
) -> list[str]:
    """
    Extract dominant words and bigrams for one pain-point group.
    """
    cluster_matrix = tfidf_matrix[
        labels == pain_point_id
    ]

    mean_weights = np.asarray(
        cluster_matrix.mean(axis=0)
    ).ravel()

    indices = mean_weights.argsort()[
        ::-1
    ][:NUMBER_OF_KEYWORDS]

    return [
        feature_names[index]
        for index in indices
        if mean_weights[index] > 0
    ]


def find_representative_reviews(
    reviews: pd.DataFrame,
    tfidf_matrix: object,
    labels: np.ndarray,
    model: KMeans,
    pain_point_id: int,
) -> list[str]:
    """
    Select reviews closest to the pain-point centroid.
    """
    indices = np.where(
        labels == pain_point_id
    )[0]

    cluster_matrix = tfidf_matrix[
        indices
    ]

    centroid = model.cluster_centers_[
        pain_point_id
    ].reshape(
        1,
        -1,
    )

    similarities = cosine_similarity(
        cluster_matrix,
        centroid,
    ).ravel()

    closest_positions = np.argsort(
        similarities
    )[
        ::-1
    ][:NUMBER_OF_EXAMPLE_REVIEWS]

    global_indices = indices[
        closest_positions
    ]

    examples: list[str] = []

    for index in global_indices:

        text = str(
            reviews.iloc[index][
                "review_text"
            ]
        ).replace(
            "\n",
            " "
        )

        # Avoid gigantic CSV cells.
        if len(text) > 250:
            text = (
                text[:247]
                + "..."
            )

        examples.append(
            text
        )

    return examples


def build_pain_point_summary(
    reviews: pd.DataFrame,
    tfidf_matrix: object,
    labels: np.ndarray,
    model: KMeans,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """
    Create a human-readable summary for every complaint cluster.
    """
    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    records: list[
        dict[str, object]
    ] = []

    total_reviews = len(
        reviews
    )

    for pain_point_id in sorted(
        np.unique(labels)
    ):

        mask = (
            labels
            == pain_point_id
        )

        cluster_reviews = reviews[
            mask
        ]

        keywords = extract_keywords(
            tfidf_matrix=tfidf_matrix,
            labels=labels,
            feature_names=feature_names,
            pain_point_id=int(
                pain_point_id
            ),
        )

        representative_reviews = (
            find_representative_reviews(
                reviews=reviews,
                tfidf_matrix=tfidf_matrix,
                labels=labels,
                model=model,
                pain_point_id=int(
                    pain_point_id
                ),
            )
        )

        review_count = len(
            cluster_reviews
        )

        records.append(
            {
                "pain_point_id": int(
                    pain_point_id
                ),
                "review_count": (
                    review_count
                ),
                "share_of_negative_reviews": round(
                    review_count
                    / total_reviews,
                    3,
                ),
                "average_rating": round(
                    cluster_reviews[
                        "review_rating"
                    ].mean(),
                    2,
                ),
                "game_count": (
                    cluster_reviews[
                        "app_id"
                    ].nunique()
                ),
                "keywords": (
                    " | ".join(
                        keywords
                    )
                ),
                "representative_reviews": (
                    " || ".join(
                        representative_reviews
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# LINK REVIEWS TO GAME CLUSTERS
# =============================================================================

def attach_game_clusters(
    reviews: pd.DataFrame,
    game_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the existing game cluster ID to every collected review.
    """
    game_cluster_mapping = (
        game_clusters[
            [
                "app_id",
                "cluster_id",
            ]
        ]
        .drop_duplicates(
            subset=["app_id"]
        )
    )

    merged = reviews.merge(
        game_cluster_mapping,
        on="app_id",
        how="left",
        validate="many_to_one",
    )

    return merged


def build_cluster_pain_points(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the pain-point distribution inside each game segment.

    Only clusters with enough collected negative reviews are considered
    reliable enough for analysis.
    """
    usable_reviews = reviews.dropna(
        subset=[
            "cluster_id",
            "pain_point_id",
        ]
    ).copy()

    if usable_reviews.empty:
        return pd.DataFrame()

    usable_reviews[
        "cluster_id"
    ] = usable_reviews[
        "cluster_id"
    ].astype(int)

    grouped = (
        usable_reviews
        .groupby(
            [
                "cluster_id",
                "pain_point_id",
            ]
        )
        .size()
        .reset_index(
            name="review_count"
        )
    )

    cluster_totals = (
        usable_reviews
        .groupby(
            "cluster_id"
        )
        .size()
        .reset_index(
            name=(
                "cluster_negative_reviews"
            )
        )
    )

    grouped = grouped.merge(
        cluster_totals,
        on="cluster_id",
        how="left",
    )

    grouped[
        "pain_point_share"
    ] = (
        grouped[
            "review_count"
        ]
        / grouped[
            "cluster_negative_reviews"
        ]
    ).round(3)

    grouped[
        "sufficient_review_coverage"
    ] = (
        grouped[
            "cluster_negative_reviews"
        ]
        >= MIN_NEGATIVE_REVIEWS_FOR_CLUSTER_ANALYSIS
    )

    return grouped.sort_values(
        [
            "cluster_id",
            "pain_point_share",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_summary(
    pain_point_summary: pd.DataFrame,
) -> None:
    """
    Print the discovered pain-point groups.
    """
    print("\n")
    print("=" * 75)
    print("PLAYER PAIN POINTS")
    print("=" * 75)

    for _, pain_point in (
        pain_point_summary.iterrows()
    ):

        print(
            f"\nPain point "
            f"{int(pain_point['pain_point_id'])}"
        )

        print(
            f"Reviews: "
            f"{int(pain_point['review_count'])}"
        )

        print(
            f"Share of negative reviews: "
            f"{pain_point['share_of_negative_reviews']:.1%}"
        )

        print(
            f"Games represented: "
            f"{int(pain_point['game_count'])}"
        )

        print(
            f"Keywords: "
            f"{pain_point['keywords']}"
        )

        print(
            "Representative reviews:"
        )

        examples = str(
            pain_point[
                "representative_reviews"
            ]
        ).split(
            " || "
        )

        for example in examples:
            print(
                f'  - "{example}"'
            )

        print(
            "-" * 75
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Run the complete negative-review pain-point pipeline.
    """
    print(
        "Loading App Store reviews..."
    )

    reviews = load_reviews()

    print(
        f"Total collected reviews: "
        f"{len(reviews)}"
    )

    negative_reviews = (
        prepare_negative_reviews(
            reviews
        )
    )

    print(
        f"Usable 1-2 star reviews: "
        f"{len(negative_reviews)}"
    )

    print(
        f"Games represented: "
        f"{negative_reviews['app_id'].nunique()}"
    )

    print(
        "\nBuilding review TF-IDF features..."
    )

    (
        vectorizer,
        tfidf_matrix,
    ) = create_tfidf_features(
        negative_reviews
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    evaluation = (
        evaluate_cluster_counts(
            tfidf_matrix
        )
    )

    selected_k = (
        select_cluster_count(
            evaluation
        )
    )

    print(
        "\nTraining final pain-point model..."
    )

    model, labels = (
        train_pain_point_model(
            tfidf_matrix=tfidf_matrix,
            number_of_clusters=selected_k,
        )
    )

    negative_reviews[
        "pain_point_id"
    ] = labels

    pain_point_summary = (
        build_pain_point_summary(
            reviews=negative_reviews,
            tfidf_matrix=tfidf_matrix,
            labels=labels,
            model=model,
            vectorizer=vectorizer,
        )
    )

    print(
        "\nLinking reviews to game segments..."
    )

    game_clusters = (
        load_game_clusters()
    )

    negative_reviews = (
        attach_game_clusters(
            reviews=negative_reviews,
            game_clusters=game_clusters,
        )
    )

    cluster_pain_points = (
        build_cluster_pain_points(
            negative_reviews
        )
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    negative_reviews.to_csv(
        PAIN_POINT_REVIEWS_FILE,
        index=False,
        encoding="utf-8",
    )

    pain_point_summary.to_csv(
        PAIN_POINT_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    cluster_pain_points.to_csv(
        CLUSTER_PAIN_POINTS_FILE,
        index=False,
        encoding="utf-8",
    )

    evaluation.to_csv(
        PAIN_POINT_EVALUATION_FILE,
        index=False,
        encoding="utf-8",
    )

    display_summary(
        pain_point_summary
    )

    print(
        "\nPain-point analysis completed successfully."
    )

    print(
        f"\nPain-point summary:\n"
        f"{PAIN_POINT_SUMMARY_FILE}"
    )

    print(
        f"\nCluster pain points:\n"
        f"{CLUSTER_PAIN_POINTS_FILE}"
    )


if __name__ == "__main__":
    main()