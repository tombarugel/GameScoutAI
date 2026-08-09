"""
Build semantic clusters of App Store games from their descriptions.

Pipeline:
1. Load the enriched App Store dataset.
2. Prepare a text representation for each game.
3. Convert text into TF-IDF features.
4. Reduce dimensionality with Truncated SVD.
5. evaluate several KMeans configurations.
6. Select the number of clusters with the best silhouette score.
7. Extract representative keywords and games for every cluster.
8. Save the clustered dataset and cluster summary.

Run from the project root with:

    python scripts/build_game_clusters.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from src.text_cleaning import (
    clean_game_description,
    normalize_genres,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Numbers of clusters that will be tested.
MIN_CLUSTERS = 3
MAX_CLUSTERS = 20

# Number of semantic dimensions retained after SVD.
# The script automatically reduces this value if the dataset is too small.
TARGET_SVD_COMPONENTS = 50

# Number of keywords displayed for each cluster.
NUMBER_OF_KEYWORDS = 12

# Number of representative games displayed for each cluster.
NUMBER_OF_EXAMPLE_GAMES = 5


PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "app_store_games_enriched.csv"
)

CLUSTERED_GAMES_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "app_store_games_clustered.csv"
)

CLUSTER_SUMMARY_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "cluster_summary.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_enriched_data() -> pd.DataFrame:
    """
    Load and validate the enriched App Store dataset.

    Returns:
        Cleaned DataFrame containing games with usable descriptions.

    Raises:
        FileNotFoundError:
            If the enriched CSV does not exist.
        ValueError:
            If required columns are missing or no descriptions are available.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The enriched dataset was not found. "
            "Run scripts/enrich_app_store_metadata.py first."
        )

    dataframe = pd.read_csv(INPUT_FILE)

    required_columns = {
        "app_id",
        "title",
        "developer",
        "description",
        "chart_type",
        "chart_position",
        "average_rating",
        "rating_count",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "The enriched dataset is missing required columns: "
            f"{missing_text}"
        )

    dataframe = dataframe.copy()

    dataframe["description"] = (
        dataframe["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Descriptions that are too short carry little useful semantic information.
    dataframe = dataframe[
        dataframe["description"].str.len() >= 50
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No sufficiently detailed game descriptions were found."
        )

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    dataframe["average_rating"] = pd.to_numeric(
        dataframe["average_rating"],
        errors="coerce",
    )

    dataframe["rating_count"] = pd.to_numeric(
        dataframe["rating_count"],
        errors="coerce",
    )

    dataframe["chart_position"] = pd.to_numeric(
        dataframe["chart_position"],
        errors="coerce",
    )

    dataframe = dataframe.reset_index(drop=True)

    return dataframe


# =============================================================================
# TEXT PREPARATION
# =============================================================================

def build_text_corpus(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Build one cleaned semantic document per game.

    The title is deliberately excluded because franchise names and repeated
    naming conventions tend to group games by publisher rather than by actual
    gameplay mechanics.

    The corpus combines:
    - cleaned descriptions;
    - normalized structured genres with moderate additional weight.
    """
    if "detailed_genres" in dataframe.columns:
        raw_genres = dataframe[
            "detailed_genres"
        ]

    elif "genres" in dataframe.columns:
        raw_genres = dataframe[
            "genres"
        ]

    else:
        raw_genres = pd.Series(
            "",
            index=dataframe.index,
        )

    cleaned_descriptions = dataframe[
        "description"
    ].apply(
        clean_game_description
    )

    normalized_genre_text = raw_genres.apply(
        normalize_genres
    )

    corpus = (
        cleaned_descriptions
        + " "
        + normalized_genre_text
    ).str.strip()

    if corpus.str.len().lt(20).all():
        raise ValueError(
            "Text cleaning removed too much information "
            "from the descriptions."
        )

    return corpus


def create_tfidf_features(
    corpus: pd.Series,
) -> tuple[TfidfVectorizer, object]:
    """
    Transform cleaned descriptions into TF-IDF features.

    The configuration keeps words and two-word expressions while applying
    stronger filtering than the first clustering version.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),

        # A term must appear in at least three games. This removes highly
        # specific marketing language and isolated proper nouns.
        min_df=3,

        # Terms appearing in more than 70% of games are insufficiently
        # discriminating.
        max_df=0.70,

        max_features=6000,
        sublinear_tf=True,

        # Avoid extremely long features generated from malformed text.
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )

    tfidf_matrix = vectorizer.fit_transform(
        corpus
    )

    if tfidf_matrix.shape[1] < 10:
        raise ValueError(
            "TF-IDF generated too few usable features "
            "after text cleaning."
        )

    return vectorizer, tfidf_matrix


# =============================================================================
# DIMENSION REDUCTION
# =============================================================================

def reduce_dimensions(
    tfidf_matrix: object,
) -> tuple[TruncatedSVD, Normalizer, np.ndarray]:
    """
    Compress the TF-IDF matrix using Truncated SVD.

    The reduced vectors are normalized afterwards so that clustering focuses
    on semantic direction rather than document length.
    """
    maximum_components = min(
        TARGET_SVD_COMPONENTS,
        tfidf_matrix.shape[0] - 1,
        tfidf_matrix.shape[1] - 1,
    )

    number_of_components = max(
        2,
        maximum_components,
    )

    svd = TruncatedSVD(
        n_components=number_of_components,
        random_state=RANDOM_STATE,
    )

    reduced_matrix = svd.fit_transform(
        tfidf_matrix
    )

    normalizer = Normalizer(
        copy=False
    )

    normalized_matrix = normalizer.fit_transform(
        reduced_matrix
    )

    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        f"SVD components: {number_of_components}"
    )

    print(
        "Explained variance retained: "
        f"{explained_variance:.2%}"
    )

    return (
        svd,
        normalizer,
        normalized_matrix,
    )


# =============================================================================
# CLUSTER SELECTION
# =============================================================================

def evaluate_cluster_counts(
    feature_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Train several KMeans models and evaluate them with silhouette score.

    A larger silhouette score indicates that:
    - games inside a cluster are relatively similar;
    - different clusters are relatively distinct.
    """
    maximum_valid_clusters = min(
        MAX_CLUSTERS,
        len(feature_matrix) - 1,
    )

    candidate_clusters = range(
        MIN_CLUSTERS,
        maximum_valid_clusters + 1,
    )

    evaluation_records: list[dict[str, float]] = []

    print("\nEvaluating candidate cluster counts:")

    for number_of_clusters in candidate_clusters:
        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(
            feature_matrix
        )

        score = silhouette_score(
            feature_matrix,
            labels,
            metric="euclidean",
        )

        evaluation_records.append(
            {
                "number_of_clusters": number_of_clusters,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )

        print(
            f"  k={number_of_clusters:2d} "
            f"| silhouette={score:.4f} "
            f"| inertia={model.inertia_:.2f}"
        )

    evaluation_dataframe = pd.DataFrame(
        evaluation_records
    )

    return evaluation_dataframe


def select_best_cluster_count(
    evaluation_dataframe: pd.DataFrame,
) -> int:
    """
    Select a good compromise between separation and interpretability.

    Instead of always taking the absolute maximum silhouette score, the
    function selects the smallest cluster count whose score reaches at least
    95% of the best observed score. This avoids unnecessary fragmentation.
    """
    best_score = float(
        evaluation_dataframe[
            "silhouette_score"
        ].max()
    )

    acceptable_threshold = (
        0.95 * best_score
    )

    acceptable_models = (
        evaluation_dataframe[
            evaluation_dataframe[
                "silhouette_score"
            ] >= acceptable_threshold
        ]
        .sort_values(
            "number_of_clusters"
        )
    )

    selected_row = acceptable_models.iloc[0]

    selected_cluster_count = int(
        selected_row["number_of_clusters"]
    )

    selected_score = float(
        selected_row["silhouette_score"]
    )

    print(
        "\nMaximum silhouette score: "
        f"{best_score:.4f}"
    )

    print(
        "Selected cluster count: "
        f"{selected_cluster_count}"
    )

    print(
        "Selected silhouette score: "
        f"{selected_score:.4f}"
    )

    return selected_cluster_count


def train_final_cluster_model(
    feature_matrix: np.ndarray,
    number_of_clusters: int,
) -> tuple[KMeans, np.ndarray]:
    """
    Train the final KMeans model with the selected number of clusters.
    """
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=30,
    )

    labels = model.fit_predict(
        feature_matrix
    )

    return model, labels


# =============================================================================
# CLUSTER INTERPRETATION
# =============================================================================

def extract_cluster_keywords(
    tfidf_matrix: object,
    labels: np.ndarray,
    feature_names: np.ndarray,
    cluster_id: int,
    number_of_keywords: int,
) -> list[str]:
    """
    Extract the terms with the highest average TF-IDF weight in a cluster.
    """
    cluster_mask = labels == cluster_id

    cluster_matrix = tfidf_matrix[
        cluster_mask
    ]

    mean_term_weights = np.asarray(
        cluster_matrix.mean(axis=0)
    ).ravel()

    top_indices = mean_term_weights.argsort()[
        ::-1
    ][:number_of_keywords]

    keywords = [
        feature_names[index]
        for index in top_indices
        if mean_term_weights[index] > 0
    ]

    return keywords


def calculate_cluster_ranking_score(
    cluster_dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Build an internal score used only to select representative games.

    High-ranking games, games with many ratings and highly rated games receive
    more weight. This score is not yet the final Opportunity Score.
    """
    ranking_component = (
        101
        - cluster_dataframe[
            "chart_position"
        ].fillna(100)
    ) / 100

    rating_component = (
        cluster_dataframe[
            "average_rating"
        ].fillna(0)
        / 5
    )

    rating_count_component = np.log1p(
        cluster_dataframe[
            "rating_count"
        ].fillna(0)
    )

    maximum_log_count = (
        rating_count_component.max()
    )

    if maximum_log_count > 0:
        rating_count_component = (
            rating_count_component
            / maximum_log_count
        )

    return (
        0.45 * ranking_component
        + 0.30 * rating_component
        + 0.25 * rating_count_component
    )


def build_cluster_summary(
    dataframe: pd.DataFrame,
    tfidf_matrix: object,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """
    Create an interpretable summary for every cluster.
    """
    feature_names = (
        vectorizer.get_feature_names_out()
    )

    summary_records: list[dict[str, object]] = []

    for cluster_id in sorted(
        np.unique(labels)
    ):
        cluster_dataframe = dataframe[
            dataframe["cluster_id"]
            == cluster_id
        ].copy()

        keywords = extract_cluster_keywords(
            tfidf_matrix=tfidf_matrix,
            labels=labels,
            feature_names=feature_names,
            cluster_id=cluster_id,
            number_of_keywords=NUMBER_OF_KEYWORDS,
        )

        cluster_dataframe[
            "representative_score"
        ] = calculate_cluster_ranking_score(
            cluster_dataframe
        )

        representative_games = (
            cluster_dataframe
            .sort_values(
                "representative_score",
                ascending=False,
            )
            .head(NUMBER_OF_EXAMPLE_GAMES)
            ["title"]
            .astype(str)
            .tolist()
        )

        average_rating = (
            cluster_dataframe[
                "average_rating"
            ].mean()
        )

        median_rating_count = (
            cluster_dataframe[
                "rating_count"
            ].median()
        )

        average_chart_position = (
            cluster_dataframe[
                "chart_position"
            ].mean()
        )

        developer_count = (
            cluster_dataframe[
                "developer"
            ].nunique()
        )

        free_game_share = (
            cluster_dataframe[
                "chart_type"
            ].eq("top_free")
            .mean()
        )

        summary_records.append(
            {
                "cluster_id": int(cluster_id),
                "game_count": len(
                    cluster_dataframe
                ),
                "developer_count": developer_count,
                "average_rating": round(
                    average_rating,
                    3,
                ),
                "median_rating_count": round(
                    median_rating_count,
                    0,
                ),
                "average_chart_position": round(
                    average_chart_position,
                    2,
                ),
                "free_game_share": round(
                    free_game_share,
                    3,
                ),
                "keywords": " | ".join(
                    keywords
                ),
                "representative_games": " | ".join(
                    representative_games
                ),
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_records
    )

    return summary_dataframe


# =============================================================================
# SAVING
# =============================================================================

def save_results(
    clustered_dataframe: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    evaluation_dataframe: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    normalizer: Normalizer,
    kmeans_model: KMeans,
) -> None:
    """
    Save datasets, evaluation results and trained models.
    """
    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustered_dataframe.to_csv(
        CLUSTERED_GAMES_FILE,
        index=False,
        encoding="utf-8",
    )

    cluster_summary.to_csv(
        CLUSTER_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    evaluation_dataframe.to_csv(
        PROCESSED_DATA_DIRECTORY
        / "cluster_evaluation.csv",
        index=False,
        encoding="utf-8",
    )

    joblib.dump(
        vectorizer,
        MODEL_DIRECTORY / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        svd,
        MODEL_DIRECTORY / "svd_model.joblib",
    )

    joblib.dump(
        normalizer,
        MODEL_DIRECTORY / "normalizer.joblib",
    )

    joblib.dump(
        kmeans_model,
        MODEL_DIRECTORY / "kmeans_model.joblib",
    )


# =============================================================================
# DISPLAY
# =============================================================================

def display_cluster_summary(
    cluster_summary: pd.DataFrame,
) -> None:
    """
    Print an interpretable cluster overview in the terminal.
    """
    print("\nCluster summary")
    print("===============\n")

    for _, cluster in cluster_summary.iterrows():
        print(
            f"Cluster {int(cluster['cluster_id'])}"
        )

        print(
            f"Games: {int(cluster['game_count'])}"
        )

        print(
            f"Keywords: {cluster['keywords']}"
        )

        print(
            "Representative games: "
            f"{cluster['representative_games']}"
        )

        print(
            "Average rating: "
            f"{cluster['average_rating']}"
        )

        print(
            "Average chart position: "
            f"{cluster['average_chart_position']}"
        )

        print("-" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Run the complete semantic clustering pipeline.
    """
    print("Loading enriched App Store data...")

    dataframe = load_enriched_data()

    print(
        f"Games with usable descriptions: "
        f"{len(dataframe)}"
    )

    corpus = build_text_corpus(
        dataframe
    )

    dataframe["cleaned_text"] = corpus

    print("\nBuilding TF-IDF features...")

    vectorizer, tfidf_matrix = (
        create_tfidf_features(
            corpus
        )
    )

    print(
        "TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print("\nReducing dimensions with SVD...")

    (
        svd,
        normalizer,
        feature_matrix,
    ) = reduce_dimensions(
        tfidf_matrix
    )

    evaluation_dataframe = (
        evaluate_cluster_counts(
            feature_matrix
        )
    )

    best_cluster_count = (
        select_best_cluster_count(
            evaluation_dataframe
        )
    )

    print("\nTraining final KMeans model...")

    kmeans_model, labels = (
        train_final_cluster_model(
            feature_matrix=feature_matrix,
            number_of_clusters=best_cluster_count,
        )
    )

    dataframe["cluster_id"] = labels

    cluster_summary = build_cluster_summary(
        dataframe=dataframe,
        tfidf_matrix=tfidf_matrix,
        labels=labels,
        vectorizer=vectorizer,
    )

    save_results(
        clustered_dataframe=dataframe,
        cluster_summary=cluster_summary,
        evaluation_dataframe=evaluation_dataframe,
        vectorizer=vectorizer,
        svd=svd,
        normalizer=normalizer,
        kmeans_model=kmeans_model,
    )

    display_cluster_summary(
        cluster_summary
    )

    print("\nClustering completed successfully.")

    print(
        f"Clustered games saved to: "
        f"{CLUSTERED_GAMES_FILE}"
    )

    print(
        f"Cluster summary saved to: "
        f"{CLUSTER_SUMMARY_FILE}"
    )

    print(
        f"Models saved to: "
        f"{MODEL_DIRECTORY}"
    )


if __name__ == "__main__":
    main()