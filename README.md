# GameScout AI

GameScout AI is a compact decision-support tool for mobile game designers. It combines a current US App Store Games snapshot, metadata enrichment, unsupervised NLP clustering, an interpretable opportunity score, public review mining, and live generative AI concept creation.

## Evaluator quick start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The application ships with cached CSV snapshots in `data/processed/`, so the market dashboard, segments, opportunity ranking and review evidence work immediately and reproducibly.

### Live AI generation

The app calls a Cloudflare Worker using Workers AI. The default production URL is configured in `src/ai_client.py` and can be overridden without editing code:

```bash
export GAMESCOUT_WORKER_URL="https://your-worker.workers.dev"
streamlit run app.py
```

For local Worker development:

```bash
cd cloudflare-worker
npx wrangler dev
```

Then in another terminal:

```bash
export GAMESCOUT_WORKER_URL="http://localhost:8787"
streamlit run app.py
```

Deploy the Worker once with:

```bash
cd cloudflare-worker
npx wrangler deploy
```

The evaluator does **not** need an API key. The AI binding is held server-side by Cloudflare.

## Product flow

1. **Market Overview** — current Top Free / Top Paid Games and developer presence.
2. **Game Segments** — TF-IDF → SVD → KMeans semantic segments.
3. **Opportunities** — transparent scoring based on market strength, player validation, scarcity, developer diversity, semantic coherence and confidence.
4. **Concept Generator** — market evidence + available review pain points → Cloudflare Workers AI → structured game concept.

## Refreshing the data

The shipped snapshot is intentionally cached for evaluator reliability. To refresh the pipeline, run the scripts from the project root in this order:

```bash
python scripts/collect_app_store_data.py
python scripts/enrich_app_store_metadata.py
python scripts/build_game_clusters.py
python scripts/build_opportunity_scores.py
python scripts/collect_app_store_reviews.py
python scripts/build_review_pain_points.py
```

Public App Store review availability is incomplete and varies by app. Review evidence is therefore treated as an optional enrichment signal rather than a required input.

## Important limitations

- The current data are a snapshot, so the opportunity score measures **current market signals**, not historical growth.
- Unsupervised clusters are exploratory; franchise-dominated and mixed segments remain visible rather than being hidden.
- Public review feeds have uneven coverage.
- The Opportunity Score is an interpretable heuristic, not a revenue or download forecast.
- AI-generated concepts are creative suggestions grounded in supplied evidence, not predictions of commercial success.

## Live evaluator flow

The application now opens on an analysis launcher instead of a pre-filled dashboard.

- **Start Live Analysis** executes the real project scripts in order from current App Store data through review pain-point analysis.
- **Use Cached Snapshot** is a fallback for a fast/reproducible demo when an external Apple endpoint is unavailable.
- After completion, the sidebar exposes the market overview, ML segments, opportunity ranking and AI concept generator.

For the complete live flow, keep internet access available. Review collection uses Apple's public review feed and can take several minutes because coverage varies by app.
