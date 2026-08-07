# Experimental semantic-embedding baseline

This folder contains the SentenceTransformer clustering experiment used to compare the production TF-IDF/SVD/KMeans pipeline with a transformer-embedding approach.

It is **not required** to run the application.

Install the optional dependency only if you want to reproduce the experiment:

```bash
python -m pip install sentence-transformers
python experiments/build_game_clusters_embeddings.py
```

The experiment was retained for methodological transparency, while the production pipeline uses TF-IDF/SVD/KMeans because it produced more actionable gameplay-oriented segments on this dataset.
