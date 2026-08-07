"""Run the GameScout AI data pipeline in order.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-reviews

The review stage is optional because Apple's public RSS review availability is
uneven and the collection can take several minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_STEPS = [
    "scripts/collect_app_store_data.py",
    "scripts/enrich_app_store_metadata.py",
    "scripts/build_game_clusters.py",
    "scripts/build_opportunity_scores.py",
]

REVIEW_STEPS = [
    "scripts/collect_app_store_reviews.py",
    "scripts/build_review_pain_points.py",
]


def run_step(relative_path: str) -> None:
    print("\n" + "=" * 78)
    print(f"Running {relative_path}")
    print("=" * 78)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-reviews",
        action="store_true",
        help="Refresh market data and opportunity scores without collecting reviews.",
    )
    args = parser.parse_args()

    steps = BASE_STEPS + ([] if args.skip_reviews else REVIEW_STEPS)

    for step in steps:
        run_step(step)

    print("\nPipeline completed successfully.")
    print("Launch the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
