# 🎮 GameScout AI

> **AI-powered market intelligence platform for mobile game discovery and concept generation.**

GameScout AI is an end-to-end market intelligence platform designed to help game studios discover market opportunities and generate original game concepts using real App Store data and AI.

Instead of relying on static dashboards or manual market research, the platform automatically:

- 📈 Collects live App Store rankings
- 🧩 Enriches game metadata
- 🔍 Discovers semantic market segments
- 🎯 Identifies underserved opportunities
- 💬 Analyzes thousands of player reviews
- 🤖 Generates AI-assisted game concepts grounded in market evidence

---

# 🚀 Live Demo

```
https://gamescoutai-qjstw95hcerp2qem2tapmz.streamlit.app/
```

No installation is required.

---

# 📸 Application Overview

*(Add one or two screenshots of the application here.)*

Suggested screenshots:

- Home page
- Opportunity Dashboard
- AI Concept Generator

---

# ✨ Features

- ✅ Live App Store scraping
- ✅ Metadata enrichment
- ✅ Automatic market segmentation
- ✅ Opportunity scoring
- ✅ Player review analysis
- ✅ Pain-point extraction
- ✅ AI game concept generation
- ✅ Interactive Streamlit interface

---

# 🏗 Architecture

```text
                 App Store
                     │
                     ▼
          Live Game Rankings
                     │
                     ▼
         Metadata Enrichment
                     │
                     ▼
            Text Cleaning
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
          Opportunity Engine
                     │
                     ▼
         Review Collection
                     │
                     ▼
       Pain Point Extraction
                     │
                     ▼
        Cloudflare Workers AI
                     │
                     ▼
          Game Concept Generator
```

---

# ⚙️ Technology Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | scikit-learn |
| NLP | TF-IDF + Truncated SVD |
| Clustering | KMeans |
| Data Sources | App Store RSS + iTunes API |
| AI | Cloudflare Workers AI |
| Deployment | Streamlit Community Cloud + Cloudflare Workers |

---

# 🚀 Try GameScout AI

The easiest way to experience GameScout AI is through the public web application.

## Live Demo

👉 **https://gamescout-ai.streamlit.app**

Everything is already deployed and configured.

- ✅ Streamlit frontend
- ✅ Cloudflare Worker
- ✅ Workers AI backend

**No installation, terminal, or configuration is required.**

---

# 👨‍💻 Local Development

The following instructions are only required if you want to modify or extend the project.

## Clone the repository

```bash
git clone https://github.com/tombarugel/GameScoutAI.git

cd GameScoutAI
```

## Install Python dependencies

```bash
pip install -r requirements.txt
```

## Launch the application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# ☁️ Deploy Your Own AI Backend (Optional)

GameScout AI already uses a deployed Cloudflare Worker.

You only need the following steps if you want to deploy **your own** Worker instead of using the hosted one.

```bash
cd cloudflare-worker

npm install

npx wrangler deploy
```

Once deployed, replace the Worker URL in:

```
src/ai_client.py
```

with your own Cloudflare Workers URL.

# 🧠 Key Product Decisions

### Live market data

Rather than relying on a static dataset, the application retrieves the latest App Store rankings to provide up-to-date market intelligence.

---

### Lightweight NLP pipeline

TF-IDF combined with Truncated SVD was selected because it provides interpretable market segments while remaining lightweight and fast enough to run interactively.

Although embedding-based approaches were explored during experimentation, the TF-IDF pipeline offered a better trade-off between interpretability, reproducibility and execution time.

---

### Cloudflare Workers AI

Workers AI allows AI inference without requiring users to provide their own API key.

This makes the application significantly easier to deploy and demonstrate while keeping infrastructure lightweight.

---

### Cached snapshots

A cached mode is available to guarantee a smooth demonstration even if the App Store API is temporarily unavailable.

---

# ⚙️ Key Technical Decisions

The project is intentionally organized as a modular pipeline.

Each processing stage is isolated inside an independent Python script:

- App Store collection
- Metadata enrichment
- Clustering
- Opportunity scoring
- Review analysis
- AI generation

Intermediate results are stored as CSV files.

This architecture makes the pipeline easy to debug, rerun and extend.

The frontend (Streamlit) remains completely stateless and simply orchestrates the pipeline.

---

# ⚠️ Known Limitations

Current limitations include:

- Only the US App Store is supported.
- KMeans requires choosing the number of clusters beforehand.
- Review availability depends on Apple's public endpoints.
- Opportunity scores are based on heuristic rules rather than predictive models.
- AI-generated concepts are intended for ideation and not as production-ready game designs.

---

# 🚀 What I Would Build Next

With one additional week, I would focus on the following improvements.

### Multi-country analysis

Compare markets across the US, Japan, South Korea and China.

---

### Historical market tracking

Monitor ranking evolution over time to detect emerging genres.

---

### Smarter opportunity scoring

Replace part of the heuristic scoring engine with an LLM-assisted reasoning layer.

---

### Competitor evolution

Track how individual genres evolve week after week.

---

### Agentic research assistant

Allow users to ask questions such as:

> "Generate three game concepts targeting women aged 35–50 based on the latest merge-game trends."

The assistant would automatically retrieve relevant market data before generating tailored recommendations.

---

# 📂 Repository Structure

```
GameScoutAI/

│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
├── scripts/
├── data/
├── models/
│
└── cloudflare-worker/
```

---

# 👨‍💻 Author

**Tom Barugel**

Dual Degree

- Georgia Institute of Technology
- Arts et Métiers ParisTech

---

# 📄 License

This repository was developed as part of a technical case study.

It is intended for demonstration and evaluation purposes.