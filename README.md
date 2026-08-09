# 🎮 GameScout AI

**AI-powered market intelligence for mobile game discovery and concept generation.**

GameScout AI is an end-to-end prototype designed to turn live App Store data into actionable game opportunities.

The pipeline collects current market data, groups games into semantic segments, analyzes player feedback, ranks potential opportunities, and finally uses generative AI to propose new game concepts grounded in those market signals.

---

## 🚀 Live Application

The project is already deployed and can be used directly from a browser:

**https://gamescout-ai.streamlit.app**

No installation, terminal, API key, or Cloudflare configuration is required.

The deployed application includes both:

- the Streamlit application;
- the deployed Cloudflare Workers AI backend.

The application provides two ways to explore the pipeline:

### Live Analysis

Runs the pipeline from fresh App Store data:

**App Store collection → metadata enrichment → clustering → opportunity scoring → review collection → pain-point analysis → AI concept generation**

Review collection and pain-point analysis are always included in the live pipeline.

The user can limit the maximum number of reviews collected per game to control execution time.

### Cached Snapshot

Loads the latest processed dataset stored in the repository.

This provides a fast and reliable way to explore the application without rerunning data collection and machine-learning steps.

The snapshot represents the results of a previous pipeline run. Therefore, changes to clustering or scoring parameters only appear in Cached Snapshot after a new analysis has been run and the processed files have been updated.

---

# 👨‍💻 Run Locally

To run the application from the source code:

```bash
git clone https://github.com/tombarugel/GameScoutAI.git
cd GameScoutAI

pip install -r requirements.txt
streamlit run app.py
```

The local application uses the already deployed AI backend, so no additional Cloudflare setup is required.

---

# 🏗️ Pipeline Architecture

```text
                 App Store
                     │
                     ▼
            Current Rankings
                     │
                     ▼
          Metadata Enrichment
                     │
                     ▼
        Game Description Cleaning
                     │
                     ▼
                  TF-IDF
                     │
                     ▼
             Truncated SVD
                     │
                     ▼
                 KMeans
                     │
                     ▼
        Semantic Market Segments
                     │
                     ▼
          Opportunity Scoring
                     │
                     ▼
            Review Collection
                     │
                     ▼
         Pain-Point Extraction
                     │
                     ▼
          Selected Opportunity
                     │
                     ▼
         Cloudflare Workers AI
                     │
                     ▼
          New Game Concept
```

---

# 🧠 Key Product & Technical Decisions

## 1. Semantic Market Segmentation

The App Store's predefined categories are too broad to represent specific gameplay propositions.

GameScout therefore creates its own market segmentation using the textual descriptions of the games.

The production pipeline uses:

1. text cleaning;
2. **TF-IDF** using unigrams and bigrams;
3. **Truncated SVD** for dimensionality reduction;
4. vector normalization;
5. **KMeans** clustering.

This approach was selected because it is lightweight, reproducible and sufficiently interpretable for an exploratory product tool.

An embedding-based clustering approach was also explored during development, but the TF-IDF/SVD pipeline was retained as the production approach.

---

## 2. How the Number of Clusters Is Selected

The number of clusters is **not manually fixed to 12, 18 or 20**.

For each analysis, GameScout evaluates several candidate values of `k`.

The current search range is:

```text
k = 3 ... 20
```

For each value, a KMeans model is fitted and evaluated using the **silhouette score**.

The silhouette score measures whether games are:

- similar to other games inside their cluster;
- sufficiently different from games assigned to other clusters.

GameScout then identifies the best silhouette score obtained in the tested range.

Rather than automatically choosing the absolute best `k`, the pipeline selects:

> **the smallest k reaching at least 95% of the best observed silhouette score.**

The purpose of this rule is to avoid creating unnecessary additional segments when a simpler segmentation provides almost equivalent separation.

For example, if:

```text
k = 15 → 0.145
k = 16 → 0.148
k = 17 → 0.151
k = 18 → 0.152
```

95% of the best score would be approximately `0.144`.

The algorithm would therefore select `k = 15`, rather than automatically selecting 18.

The upper bound was extended from 12 to 20 during development because the initial evaluation sometimes reached its best silhouette score at the previous upper boundary. Testing a wider range makes it possible to determine whether that result represented a genuine segmentation structure or simply an artificial search limit.

The resulting clusters should be interpreted as **exploratory market segments**, not as a definitive taxonomy of mobile games.

---

## 3. Cluster Naming

KMeans only returns numerical cluster IDs.

GameScout therefore generates human-readable segment names after clustering.

Cluster naming is intentionally **deterministic rather than LLM-generated**. This keeps the analytical pipeline reproducible and prevents the market segmentation itself from depending on generative AI.

Names are derived from several signals:

- the cluster's most important TF-IDF keywords;
- representative games;
- dominant semantic themes;
- strong franchise signals when applicable.

Semantic families such as puzzle, merge, RPG, strategy, simulation, sports or management are identified using exact word matching.

For example:

```text
merge + decorate + renovation
→ Merge & Decoration Games

block + sort + color
→ Block & Sorting Puzzle

fantasy + RPG + battle
→ Fantasy RPG & Combat
```

Exact-token matching is used rather than substring matching. This prevents false classifications such as interpreting `"cards"` as containing the racing keyword `"car"`.

Strong franchise clusters can also receive explicit labels when the data clearly represents one franchise.

Finally, if two clusters receive the same initial label, additional discriminating keywords can be used to keep the displayed market segments distinct.

---

# 🎯 Opportunity Score

Once the market has been segmented, GameScout ranks the clusters using an **Opportunity Score from 0 to 100**.

The score is deliberately a **transparent heuristic decision-support metric**.

It is **not** intended to predict revenue, downloads, retention or the probability that a game will succeed.

Instead, it tries to answer:

> **Which currently visible market segments combine evidence of demand with room for additional competition?**

---

## Score Composition

The score contains six components:

| Component | Maximum | Purpose |
|---|---:|---|
| **Market Strength** | **25** | Measures current visibility/performance in App Store rankings |
| **Player Validation** | **20** | Measures whether players appear to validate games in the segment |
| **Competition Gap** | **15** | Looks for demand that is not accompanied by excessive observed supply |
| **Developer Diversity** | **15** | Distinguishes broad markets from publisher/franchise concentration |
| **Semantic Coherence** | **15** | Measures whether the cluster represents a meaningful semantic segment |
| **Confidence** | **10** | Reduces confidence in conclusions based on very small clusters |
| **Total** | **100** | |

The largest weights are assigned to **Market Strength** and **Player Validation**, because observable demand is treated as the strongest prerequisite for a market opportunity.

Competition and developer structure then help determine whether that demand appears accessible.

Semantic coherence and confidence act as safeguards against ranking poorly defined or undersampled clusters too highly.

---

## How Each Component Is Calculated

### Market Strength — 25 points

Market Strength is based on the cluster's **average current chart position**.

A cluster containing games that rank higher in the observed App Store charts receives a stronger score.

Cluster values are normalized relative to the other segments in the same market snapshot.

---

### Player Validation — 20 points

Player Validation combines:

```text
65% → average rating quality
35% → rating volume
```

Conceptually:

```text
Player Validation
= 0.65 × Rating Quality
+ 0.35 × Rating Volume
```

Rating volume is log-transformed before normalization.

This prevents extremely large games from completely dominating the metric while still retaining adoption as useful evidence.

---

### Competition Gap — 15 points

Competition Gap looks at two forms of scarcity:

```text
55% → game scarcity
45% → developer scarcity
```

Conceptually:

```text
Raw Competition Gap
= 0.55 × Game Scarcity
+ 0.45 × Developer Scarcity
```

However, **few games do not automatically mean a good opportunity**.

A cluster containing only a few games from the same publisher may represent a successful franchise rather than an underserved market.

GameScout therefore applies a concentration penalty based on the share of games belonging to the largest developer:

```text
Competition Gap
= Raw Gap × (1 - 0.80 × Top Developer Share)
```

This prevents publisher-dominated clusters from receiving an artificially strong opportunity signal simply because they contain few games.

---

### Developer Diversity — 15 points

Developer Diversity is derived from publisher concentration:

```text
Developer Diversity
= 1 - Top Developer Share
```

A segment where several developers successfully compete therefore receives a higher score than a segment almost entirely controlled by one publisher.

---

### Semantic Coherence — 15 points

A market opportunity is only useful if the underlying cluster represents a meaningful group of games.

GameScout therefore evaluates how semantically similar games are inside each cluster.

TF-IDF representations of game descriptions are compared with their cluster centroid using **cosine similarity**.

The average similarity becomes the cluster's semantic-coherence signal.

The values are then normalized across the current market snapshot.

---

### Confidence — 10 points

Very small clusters provide weaker evidence.

Confidence therefore increases with the number of games observed in the cluster:

```text
Confidence = min(Number of Games / 15, 1)
```

A segment containing at least 15 observed games receives the full confidence contribution.

---

## Final Calculation

The final score is additive:

```text
Opportunity Score
=
Market Strength
+ Player Validation
+ Competition Gap
+ Developer Diversity
+ Semantic Coherence
+ Confidence
```

with a maximum value of:

```text
100
```

Because several components are normalized relative to the current dataset, the score should primarily be used to **compare opportunities inside the same market snapshot**.

An Opportunity Score of 80 should therefore not be interpreted as an "80% probability of success."

---

# 💬 Player Review & Pain-Point Analysis

Market performance alone does not explain what players dislike about existing products.

GameScout therefore complements market-level analysis with live player reviews.

For games in the analyzed market, the pipeline:

1. collects App Store reviews;
2. identifies negative or critical feedback;
3. processes review text;
4. extracts recurring pain-point signals;
5. aggregates those signals at the market-segment level.

The maximum number of reviews collected per game can be limited in the interface to provide a trade-off between analysis depth and execution time.

These pain points are then provided as additional evidence to the AI concept-generation layer.

---

# 🤖 AI Concept Generation

Generative AI is used **after**, not before, the analytical pipeline.

The model does not decide which market segment represents the strongest opportunity.

Instead, the user first selects an opportunity identified by the deterministic data pipeline.

GameScout then sends structured evidence about that segment to the Cloudflare Workers AI backend.

The context can include:

- Opportunity Score;
- average rating;
- number of observed games;
- number of developers;
- average chart position;
- semantic keywords;
- representative games;
- identified market signals;
- available player pain points.

The model is instructed to use representative games as **market references rather than concepts to copy**, and not to invent unsupported revenue, download or growth statistics.

---

## Creative Strategy: Safe / Differentiated / Bold

Before generating the concept, the user selects how far the AI should move away from already validated market patterns.

### 🛡️ Safe

**Exploit-oriented.**

The generated concept stays relatively close to mechanics and product structures already validated within the selected segment.

The priority is:

- familiarity;
- feasibility;
- limited creative risk.

It answers:

> *What could we build while staying close to what already appears to work?*

---

### ⚖️ Differentiated

**Balanced exploration / exploitation.**

This is the default strategy.

The concept retains a validated market core while introducing a meaningful mechanic, theme or product twist.

The objective is to differentiate the product without ignoring the evidence found during market analysis.

It answers:

> *How could we enter this segment without simply reproducing an existing game?*

---

### 🚀 Bold

**Exploration-oriented.**

The market evidence remains the starting point, but the AI receives greater freedom to recombine those signals into a less conventional proposition.

The priority shifts toward originality while remaining plausibly connected to the identified opportunity.

It answers:

> *What more original concept could this market opportunity inspire?*

---

**Safe, Differentiated and Bold do not modify the clustering, market data or Opportunity Score.**

They only change the creative instruction sent to the generative model after an opportunity has already been selected.

---

# ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Application | Streamlit |
| Core pipeline | Python |
| Data manipulation | pandas / NumPy |
| Machine learning | scikit-learn |
| Text representation | TF-IDF |
| Dimensionality reduction | Truncated SVD |
| Segmentation | KMeans |
| Similarity | Cosine similarity |
| Data source | Apple App Store public endpoints |
| AI backend | Cloudflare Workers AI |
| Application hosting | Streamlit Community Cloud |
| AI hosting | Cloudflare Workers |

---

# 📂 Repository Structure

```text
GameScoutAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── ai_client.py
│   ├── app_data.py
│   ├── data_loader.py
│   ├── live_pipeline.py
│   ├── market_analysis.py
│   ├── opportunity_engine.py
│   └── text_cleaning.py
│
├── scripts/
│   ├── collect_app_store_data.py
│   ├── collect_app_store_reviews.py
│   ├── enrich_app_store_metadata.py
│   ├── build_game_clusters.py
│   ├── build_opportunity_scores.py
│   ├── build_review_pain_points.py
│   └── run_pipeline.py
│
├── experiments/
│   └── build_game_clusters_embeddings.py
│
├── models/
├── data/
│   ├── raw/
│   └── processed/
│
└── cloudflare-worker/
    └── src/
        └── index.ts
```

---

# ⚠️ Known Limitations

GameScout AI is a prototype built within a limited case-study timeframe.

The main current limitations are:

### Market coverage

The analysis currently focuses on one App Store market and therefore does not capture geographic differences in player preferences.

### Snapshot rather than historical trends

The current opportunity engine primarily evaluates the market at a point in time.

A strong current ranking does not necessarily imply sustained growth.

### Public-data limitations

The analysis is constrained by the information available through Apple's public endpoints.

It does not have access to proprietary metrics such as:

- downloads;
- revenue;
- retention;
- DAU/MAU;
- user acquisition costs.

The Opportunity Score should therefore be interpreted accordingly.

### Heuristic opportunity scoring

The score reflects an explicit product hypothesis about what constitutes an attractive market.

Its weights are interpretable but have not been statistically calibrated against historical game launches.

### Unsupervised segmentation

KMeans finds statistical structure in game descriptions but does not guarantee that every cluster corresponds perfectly to how a human product expert would define a genre.

Cluster naming is therefore a post-processing interpretation layer.

### Review sampling

The quantity and representativeness of reviews depend on what can be retrieved from Apple's public endpoints.

### Generative AI

Generated concepts are ideation outputs.

They should be treated as starting points for product exploration, not as validated game designs or forecasts of commercial performance.

---

# 🚀 What I Would Build With One Additional Week

With one additional week, I would prioritize improvements that make the opportunity signal more robust rather than simply adding more interface features.

## 1. Historical Market Tracking

The highest-priority improvement would be to collect App Store rankings automatically over time.

This would allow GameScout to distinguish:

```text
Large market
```

from:

```text
Growing market
```

and detect emerging segments before they become obvious from a single snapshot.

---

## 2. Multi-Country Analysis

I would extend data collection to several major mobile-gaming markets, starting with:

- United States;
- Japan;
- South Korea;
- selected European markets.

This would make it possible to identify opportunities that are mature in one geography but still emerging in another.

---

## 3. Opportunity-Score Validation

The current score is intentionally heuristic.

With additional time, I would run sensitivity analyses on its weights and compare historical scores with subsequent ranking evolution.

This would help determine whether signals such as competition gap or player validation actually correlate with future market momentum.

---

## 4. Improved Semantic Segmentation

I would continue comparing the current TF-IDF/SVD approach with modern embedding models.

The objective would not simply be to obtain a higher clustering metric, but to determine which representation produces the **most actionable and interpretable market segments for a game studio**.

---

## 5. Continuous Market Monitoring

Finally, I would turn the current one-shot analysis into a continuously updated system.

GameScout could automatically detect:

- newly emerging clusters;
- rapidly improving games;
- changes in player complaints;
- unusual ranking movements;
- new market gaps.

This would move the prototype from a market-analysis tool toward a lightweight **game opportunity radar**.

---

# 👨‍💻 Author

**Tom Barugel**

Dual-degree engineering background:

- Georgia Institute of Technology — M.S. Mechanical Engineering
- Arts et Métiers ParisTech — Engineering Degree

Developed as part of a technical case study.