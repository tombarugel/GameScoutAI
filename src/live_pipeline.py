"""Live pipeline execution helpers for the GameScout AI Streamlit app.

The evaluator can run the complete project from fresh App Store data directly
inside Streamlit. Each stage is executed with the same Python interpreter that
runs the app, and cached CSVs remain available as a fallback if an external
source is temporarily unavailable.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class PipelineStage:
    key: str
    title: str
    description: str
    script: str


PIPELINE_STAGES = [
    PipelineStage(
        key="collect",
        title="Collect App Store rankings",
        description="Fetch the current US Top Free and Top Paid Games charts.",
        script="scripts/collect_app_store_data.py",
    ),
    PipelineStage(
        key="enrich",
        title="Enrich game metadata",
        description="Retrieve descriptions, genres, ratings and developer metadata.",
        script="scripts/enrich_app_store_metadata.py",
    ),
    PipelineStage(
        key="cluster",
        title="Build semantic game segments",
        description="Clean text, vectorize with TF-IDF, reduce with SVD and cluster with KMeans.",
        script="scripts/build_game_clusters.py",
    ),
    PipelineStage(
        key="opportunities",
        title="Score market opportunities",
        description="Rank segments using chart strength, validation, scarcity, diversity and coherence.",
        script="scripts/build_opportunity_scores.py",
    ),
    PipelineStage(
        key="reviews",
        title="Collect public player reviews",
        description="Collect available public US App Store reviews and cache the coverage report.",
        script="scripts/collect_app_store_reviews.py",
    ),
    PipelineStage(
        key="pain_points",
        title="Discover recurring player pain points",
        description="Cluster negative reviews and connect recurring complaints back to game segments.",
        script="scripts/build_review_pain_points.py",
    ),
]


def run_stage(stage: PipelineStage,max_reviews: int | None = None,) -> Iterator[str]:
    """Run one pipeline stage and yield its combined stdout/stderr line by line."""
    script_path = PROJECT_ROOT / stage.script
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    command = [sys.executable, str(script_path)]

    # Only the review collection stage accepts this argument
    if stage.key == "reviews" and max_reviews is not None:
        command.extend([
            "--max-reviews",
            str(max_reviews),
        ])

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip("\n")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, [sys.executable, str(script_path)])


def _latest_raw_snapshot() -> Path | None:
    snapshots = list(RAW_DIR.glob("*.csv"))
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def summarize_stage(stage_key: str) -> dict[str, str]:
    """Return compact, user-facing facts produced by a completed pipeline stage."""
    if stage_key == "collect":
        snapshot = _latest_raw_snapshot()
        df = _safe_read(snapshot) if snapshot else None
        if df is None:
            return {"Status": "Snapshot created"}
        facts = {"Rows collected": f"{len(df):,}"}
        if "chart_type" in df.columns:
            free = int((df["chart_type"] == "top_free").sum())
            paid = int((df["chart_type"] == "top_paid").sum())
            facts.update({"Top Free": f"{free:,}", "Top Paid": f"{paid:,}"})
        return facts

    if stage_key == "enrich":
        df = _safe_read(PROCESSED_DIR / "app_store_games_enriched.csv")
        if df is None:
            return {"Status": "Metadata enriched"}
        facts = {"Unique games": f"{df['app_id'].nunique():,}" if "app_id" in df else f"{len(df):,}"}
        if "developer" in df:
            facts["Developers"] = f"{df['developer'].nunique():,}"
        if "description" in df:
            facts["Descriptions"] = f"{df['description'].fillna('').str.len().gt(0).sum():,}"
        return facts

    if stage_key == "cluster":
        summary = _safe_read(PROCESSED_DIR / "cluster_summary.csv")
        clustered = _safe_read(PROCESSED_DIR / "app_store_games_clustered.csv")
        facts: dict[str, str] = {}
        if summary is not None:
            facts["Segments discovered"] = f"{len(summary):,}"
        if clustered is not None:
            facts["Games segmented"] = f"{clustered['app_id'].nunique():,}" if "app_id" in clustered else f"{len(clustered):,}"
        try:
            vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
            facts["TF-IDF features"] = f"{len(vectorizer.get_feature_names_out()):,}"
        except Exception:
            pass
        try:
            svd = joblib.load(MODELS_DIR / "svd_model.joblib")
            facts["SVD dimensions"] = f"{int(svd.n_components):,}"
        except Exception:
            pass
        return facts or {"Status": "Game segments created"}

    if stage_key == "opportunities":
        df = _safe_read(PROCESSED_DIR / "opportunity_scores.csv")
        if df is None or df.empty:
            return {"Status": "Opportunity scores created"}
        score_col = "adjusted_opportunity_score" if "adjusted_opportunity_score" in df else "opportunity_score"
        top = df.sort_values(score_col, ascending=False).iloc[0]
        return {
            "Segments ranked": f"{len(df):,}",
            "Top signal": str(top.get("cluster_name", "—")),
            "Top score": f"{float(top[score_col]):.1f}/100",
        }

    if stage_key == "reviews":
        reviews = _safe_read(PROCESSED_DIR / "app_store_reviews.csv")
        coverage = _safe_read(PROCESSED_DIR / "review_collection_coverage.csv")
        facts = {}
        if reviews is not None:
            facts["Reviews collected"] = f"{len(reviews):,}"
            if "review_rating" in reviews:
                negative = pd.to_numeric(reviews["review_rating"], errors="coerce").le(2).sum()
                facts["Negative reviews"] = f"{int(negative):,}"
        if coverage is not None and "reviews_collected" in coverage:
            facts["Apps with reviews"] = f"{int(pd.to_numeric(coverage['reviews_collected'], errors='coerce').fillna(0).gt(0).sum()):,}"
        return facts or {"Status": "Review snapshot created"}

    if stage_key == "pain_points":
        summary = _safe_read(PROCESSED_DIR / "pain_point_summary.csv")
        linked = _safe_read(PROCESSED_DIR / "cluster_pain_points.csv")
        facts = {}
        if summary is not None:
            facts["Pain-point groups"] = f"{len(summary):,}"
        if linked is not None and "cluster_id" in linked:
            facts["Segments with review evidence"] = f"{linked['cluster_id'].nunique():,}"
        return facts or {"Status": "Player pain points extracted"}

    return {"Status": "Completed"}


def pipeline_snapshot_summary() -> dict[str, str]:
    """Summarize the currently cached end-to-end snapshot."""
    enriched = _safe_read(PROCESSED_DIR / "app_store_games_enriched.csv")
    opportunities = _safe_read(PROCESSED_DIR / "opportunity_scores.csv")
    reviews = _safe_read(PROCESSED_DIR / "app_store_reviews.csv")
    pain_points = _safe_read(PROCESSED_DIR / "pain_point_summary.csv")

    return {
        "Games": f"{enriched['app_id'].nunique():,}" if enriched is not None and "app_id" in enriched else "—",
        "Segments": f"{len(opportunities):,}" if opportunities is not None else "—",
        "Reviews": f"{len(reviews):,}" if reviews is not None else "—",
        "Pain points": f"{len(pain_points):,}" if pain_points is not None else "—",
    }
