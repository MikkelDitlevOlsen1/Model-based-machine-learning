# Project Plan: Movies Dataset Analysis

## Approach: Data-Driven
- [x] Problem-driven
- [x] Data-driven


## Data Source
**Kaggle: The Movies Dataset**
- URL: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- Files:

  - `movies_metadata.csv` - Movie information (title, budget, revenue, genres, release date, etc.)
  - `ratings.csv` - User ratings (userId, movieId, rating, timestamp)
  - `credits.csv` - Cast and crew information
  - `keywords.csv` - Movie keywords/tags


### Possible Research Questions:

4. How do temporal trends affect movie revenue?
5. Joint model of rating prediction and revenue estimation?

## Probabilistic Graphical Model





### Recommended Approach:
- **Start with Q4** (simpler temporal structure)
- **Then Q5** (leverage Q4 insights + shared latent variable)

## Implementation Plan
-
- [ ] Clean and preprocess data
- [ ] Define specific research question
- [ ] Formalize PGM structure
- [ ] Implement in Pyro
- [ ] Train and evaluate model
- [ ] Analyze results and insights

## Models

### Model 4 — Genre & Keywords → Revenue Distribution (Arham)

**Research question:** Can genre composition and keyword topics alone predict movie revenue, and how uncertain is that prediction?

**Approach:** Bayesian linear regression with ARD-style (Automatic Relevance Determination) group priors, implemented in Pyro with SVI (mean-field variational inference).

**Features:**
- Genre one-hot vectors (top 15 genres, weighted by 1/n_genres) — 15 features
- Keyword topics: TF-IDF on movie keywords → TruncatedSVD (20 components) — 20 features
- Log-runtime (standardised) — 1 feature

**Priors:**
```
σ_genre  ~ HalfNormal(5)     β_genre  ~ Normal(0, σ_genre)   [15 weights]
σ_kw     ~ HalfNormal(2)     β_kw     ~ Normal(0, σ_kw)      [20 weights]
β_runtime ~ Normal(0, 2)
intercept ~ Normal(14, 3)
σ_obs    ~ HalfNormal(1)
log_revenue ~ Normal(μ, σ_obs)
```

**Inflation adjustment:** Revenue and budget converted to 2024 USD using US CPI (1960–2024) before log-transform.

**Output:** Posterior predictive distribution **N(μ_genre, σ_genre²)** per movie on log-revenue scale, exported to `data/model4_posterior.csv` with precision `1/σ²` for weighted aggregation.

**Results (test set, 20% hold-out):**
- RMSE (log-revenue): **2.38**
- Mean log-likelihood: **−2.28**
- σ range: 2.08–2.52 (model self-reports uncertainty per movie)

**Notebook:** `notebooks/model4_genre_keywords.ipynb`

---

## Aggregation Plan

Product-of-experts precision-weighted fusion across all models:
```
σ_agg² = 1 / Σ_i (1/σ_i²)
μ_agg  = σ_agg² · Σ_i (μ_i / σ_i²)
```
Each model exports `(μ, σ, precision)` on the same log-revenue scale → aggregator combines them.

---

## Report Structure

1. Introduction
2. Models (one section per model — each member writes their own)
3. Aggregation
4. Discussion
5. Conclusion

**Deadline (Tuesday):** Each member should have their model explanation ready + start Aggregation section.

---

## Results & Findings
[TO BE UPDATED — populate after aggregation]
