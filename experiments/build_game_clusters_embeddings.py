"""
Experimental semantic clustering using Sentence Transformers.

This script is independent from the existing TF-IDF + SVD pipeline.

It:
1. loads the enriched App Store dataset;
2. cleans the game descriptions;
3. generates semantic embeddings with all-MiniLM-L6-v2;
4. evaluates several KMeans cluster counts;
5. selects a compact, well-performing solution;
6. identifies representative games for every cluster;
7. extracts TF-IDF keywords only to help interpret the semantic clusters;
8. saves all results under distinct filenames.

Run from the project root:

    python scripts/build_game_clusters_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable before importing modules from src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.text_cleaning import (
    clean_game_description,
    normalize_genres,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_CLUSTERS = 3
MAX_CLUSTERS = 15

NUMBER_OF_KEYWORDS = 12
NUMBER_OF_REPRESENTATIVE_GAMES = 6

PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "embeddings"

INPUT_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "app_store_games_enriched.csv"
)

CLUSTERED_GAMES_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "app_store_games_clustered_embeddings.csv"
)

CLUSTER_SUMMARY_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "cluster_summary_embeddings.csv"
)

CLUSTER_EVALUATION_FILE = (
    PROCESSED_DATA_DIRECTORY
    / "cluster_evaluation_embeddings.csv"
)

EMBEDDINGS_FILE = (
    MODEL_DIRECTORY
    / "game_embeddings.npy"
)

KMEANS_MODEL_FILE = (
    MODEL_DIRECTORY
    / "kmeans_embeddings.joblib"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_enriched_data() -> pd.DataFrame:
    """
    Load and validate the enriched App Store dataset.

    Returns:
        A cleaned DataFrame containing one row per usable chart entry.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The enriched dataset was not found. Run "
            "scripts/enrich_app_store_metadata.py first."
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
        missing_text = ", ".join(sorted(missing_columns))

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

    # Extremely short descriptions are not meaningful for semantic analysis.
    dataframe = dataframe[
        dataframe["description"].str.len() >= 50
    ].copy()

    dataframe["app_id"] = (
        dataframe["app_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    for column in [
        "chart_position",
        "average_rating",
        "rating_count",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.reset_index(drop=True)

    if dataframe.empty:
        raise ValueError(
            "No usable descriptions were found."
        )

    return dataframe


# =============================================================================
# TEXT PREPARATION
# =============================================================================

def build_semantic_corpus(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Build the semantic input for every game.

    The title is excluded to reduce franchise-name dominance.

    The corpus combines:
    - a cleaned game description;
    - normalized genre information.

    Sentence Transformers understand sentence meaning, so the text remains
    more natural than in a pure bag-of-words pipeline.
    """
    if "detailed_genres" in dataframe.columns:
        raw_genres = dataframe["detailed_genres"]

    elif "genres" in dataframe.columns:
        raw_genres = dataframe["genres"]

    else:
        raw_genres = pd.Series(
            "",
            index=dataframe.index,
        )

    cleaned_descriptions = dataframe[
        "description"
    ].apply(clean_game_description)

    normalized_genres = raw_genres.apply(
        normalize_genres
    )

    corpus = (
        cleaned_descriptions
        + " "
        + normalized_genres
    ).str.strip()

    if corpus.str.len().lt(20).any():
        short_count = int(
            corpus.str.len().lt(20).sum()
        )

        print(
            f"Warning: {short_count} games contain very little "
            "usable text after cleaning."
        )

    return corpus


# =============================================================================
# EMBEDDINGS
# =============================================================================

def create_embeddings(
    corpus: pd.Series,
) -> tuple[SentenceTransformer, np.ndarray]:
    """
    Generate one normalized semantic embedding per game.

    Normalization makes cosine similarity and Euclidean clustering more
    consistent because all vectors have unit length.
    """
    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Generating semantic embeddings...")

    embeddings = model.encode(
        corpus.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "The embedding model returned an unexpected shape."
        )

    print(
        "Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    return model, embeddings


# =============================================================================
# CLUSTER EVALUATION
# =============================================================================

def evaluate_cluster_counts(
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Test several KMeans configurations using silhouette score.
    """
    maximum_valid_clusters = min(
        MAX_CLUSTERS,
        len(embeddings) - 1,
    )

    evaluation_records: list[dict[str, float]] = []

    print("\nEvaluating semantic cluster counts:")

    for number_of_clusters in range(
        MIN_CLUSTERS,
        maximum_valid_clusters + 1,
    ):
        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=RANDOM_STATE,
            n_init=30,
        )

        labels = model.fit_predict(
            embeddings
        )

        score = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
        )

        cluster_sizes = np.bincount(labels)

        evaluation_records.append(
            {
                "number_of_clusters": number_of_clusters,
                "silhouette_score": score,
                "inertia": model.inertia_,
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
            f" | smallest={cluster_sizes.min():2d}"
            f" | largest={cluster_sizes.max():2d}"
        )

    return pd.DataFrame(
        evaluation_records
    )


def select_cluster_count(
    evaluation_dataframe: pd.DataFrame,
) -> int:
    """
    Select an interpretable model close to the maximum silhouette score.

    The smallest k reaching 95% of the best score is selected, provided its
    smallest cluster contains at least three games.

    This avoids choosing an unnecessarily fragmented solution.
    """
    sufficiently_sized = evaluation_dataframe[
        evaluation_dataframe[
            "smallest_cluster_size"
        ] >= 3
    ].copy()

    if sufficiently_sized.empty:
        sufficiently_sized = (
            evaluation_dataframe.copy()
        )

    best_score = float(
        sufficiently_sized[
            "silhouette_score"
        ].max()
    )

    threshold = 0.95 * best_score

    acceptable_models = (
        sufficiently_sized[
            sufficiently_sized[
                "silhouette_score"
            ] >= threshold
        ]
        .sort_values("number_of_clusters")
    )

    selected_row = acceptable_models.iloc[0]

    selected_cluster_count = int(
        selected_row["number_of_clusters"]
    )

    selected_score = float(
        selected_row["silhouette_score"]
    )

    print(
        "\nBest valid silhouette score: "
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


def train_final_model(
    embeddings: np.ndarray,
    number_of_clusters: int,
) -> tuple[KMeans, np.ndarray]:
    """
    Train the final KMeans model.
    """
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    labels = model.fit_predict(
        embeddings
    )

    return model, labels


# =============================================================================
# CLUSTER INTERPRETATION
# =============================================================================

def build_keyword_matrix(
    corpus: pd.Series,
) -> tuple[TfidfVectorizer, object]:
    """
    Build TF-IDF features only to name and interpret semantic clusters.

    Important:
        TF-IDF is not used to form the clusters in this pipeline.
        Clustering is performed exclusively from transformer embeddings.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,
        max_features=6000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )

    matrix = vectorizer.fit_transform(
        corpus
    )

    return vectorizer, matrix


def extract_cluster_keywords(
    tfidf_matrix: object,
    labels: np.ndarray,
    feature_names: np.ndarray,
    cluster_id: int,
) -> list[str]:
    """
    Extract terms with the highest average TF-IDF weight in a cluster.
    """
    cluster_matrix = tfidf_matrix[
        labels == cluster_id
    ]

    mean_weights = np.asarray(
        cluster_matrix.mean(axis=0)
    ).ravel()

    top_indices = mean_weights.argsort()[
        ::-1
    ][:NUMBER_OF_KEYWORDS]

    return [
        feature_names[index]
        for index in top_indices
        if mean_weights[index] > 0
    ]


def find_representative_games(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    kmeans_model: KMeans,
    cluster_id: int,
) -> list[str]:
    """
    Find games closest to the semantic center of a cluster.
    """
    cluster_indices = np.where(
        labels == cluster_id
    )[0]

    cluster_embeddings = embeddings[
        cluster_indices
    ]

    centroid = kmeans_model.cluster_centers_[
        cluster_id
    ]

    distances = np.linalg.norm(
        cluster_embeddings - centroid,
        axis=1,
    )

    closest_local_indices = np.argsort(
        distances
    )[:NUMBER_OF_REPRESENTATIVE_GAMES]

    closest_global_indices = cluster_indices[
        closest_local_indices
    ]

    return (
        dataframe.iloc[closest_global_indices][
            "title"
        ]
        .astype(str)
        .tolist()
    )


def calculate_developer_concentration(
    cluster_dataframe: pd.DataFrame,
) -> float:
    """
    Calculate the share represented by the largest developer.

    Example:
        0.70 means one developer owns 70% of the games in the cluster.
    """
    if cluster_dataframe.empty:
        return 0.0

    developer_shares = (
        cluster_dataframe[
            "developer"
        ]
        .fillna("Unknown")
        .value_counts(normalize=True)
    )

    return float(
        developer_shares.iloc[0]
    )


def build_cluster_summary(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    kmeans_model: KMeans,
    corpus: pd.Series,
) -> pd.DataFrame:
    """
    Create an interpretable summary of every semantic cluster.
    """
    vectorizer, tfidf_matrix = (
        build_keyword_matrix(corpus)
    )

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
            cluster_id=int(cluster_id),
        )

        representative_games = (
            find_representative_games(
                dataframe=dataframe,
                embeddings=embeddings,
                labels=labels,
                kmeans_model=kmeans_model,
                cluster_id=int(cluster_id),
            )
        )

        top_developer = (
            cluster_dataframe["developer"]
            .fillna("Unknown")
            .value_counts()
            .index[0]
        )

        developer_concentration = (
            calculate_developer_concentration(
                cluster_dataframe
            )
        )

        summary_records.append(
            {
                "cluster_id": int(cluster_id),
                "game_count": len(
                    cluster_dataframe
                ),
                "developer_count": (
                    cluster_dataframe[
                        "developer"
                    ].nunique()
                ),
                "top_developer": top_developer,
                "top_developer_share": round(
                    developer_concentration,
                    3,
                ),
                "average_rating": round(
                    cluster_dataframe[
                        "average_rating"
                    ].mean(),
                    3,
                ),
                "median_rating_count": round(
                    cluster_dataframe[
                        "rating_count"
                    ].median(),
                    0,
                ),
                "average_chart_position": round(
                    cluster_dataframe[
                        "chart_position"
                    ].mean(),
                    2,
                ),
                "free_game_share": round(
                    cluster_dataframe[
                        "chart_type"
                    ].eq("top_free")
                    .mean(),
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

    return pd.DataFrame(
        summary_records
    )


# =============================================================================
# SAVING AND DISPLAY
# =============================================================================

def save_results(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    evaluation_dataframe: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    kmeans_model: KMeans,
) -> None:
    """
    Save all experimental embedding results under distinct filenames.
    """
    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        CLUSTERED_GAMES_FILE,
        index=False,
        encoding="utf-8",
    )

    evaluation_dataframe.to_csv(
        CLUSTER_EVALUATION_FILE,
        index=False,
        encoding="utf-8",
    )

    cluster_summary.to_csv(
        CLUSTER_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    joblib.dump(
        kmeans_model,
        KMEANS_MODEL_FILE,
    )


def display_cluster_summary(
    cluster_summary: pd.DataFrame,
) -> None:
    """
    Print clusters in a readable terminal format.
    """
    print("\nSemantic cluster summary")
    print("========================\n")

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
            "Top developer share: "
            f"{cluster['top_developer_share']:.1%}"
        )

        print(
            "Average rating: "
            f"{cluster['average_rating']}"
        )

        print(
            "Average chart position: "
            f"{cluster['average_chart_position']}"
        )

        print("-" * 75)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Run the experimental semantic-clustering pipeline.
    """
    print("Loading enriched App Store data...")

    dataframe = load_enriched_data()

    print(
        f"Games with usable descriptions: "
        f"{len(dataframe)}"
    )

    corpus = build_semantic_corpus(
        dataframe
    )

    dataframe["embedding_input_text"] = (
        corpus
    )

    _, embeddings = create_embeddings(
        corpus
    )

    evaluation_dataframe = (
        evaluate_cluster_counts(
            embeddings
        )
    )

    selected_cluster_count = (
        select_cluster_count(
            evaluation_dataframe
        )
    )

    print("\nTraining final semantic KMeans model...")

    kmeans_model, labels = train_final_model(
        embeddings=embeddings,
        number_of_clusters=selected_cluster_count,
    )

    dataframe["cluster_id"] = labels

    cluster_summary = build_cluster_summary(
        dataframe=dataframe,
        embeddings=embeddings,
        labels=labels,
        kmeans_model=kmeans_model,
        corpus=corpus,
    )

    save_results(
        dataframe=dataframe,
        embeddings=embeddings,
        evaluation_dataframe=evaluation_dataframe,
        cluster_summary=cluster_summary,
        kmeans_model=kmeans_model,
    )

    display_cluster_summary(
        cluster_summary
    )

    print("\nSemantic clustering completed successfully.")

    print(
        f"Clustered games: {CLUSTERED_GAMES_FILE}"
    )

    print(
        f"Cluster summary: {CLUSTER_SUMMARY_FILE}"
    )

    print(
        f"Cluster evaluation: {CLUSTER_EVALUATION_FILE}"
    )


if __name__ == "__main__":
    main()