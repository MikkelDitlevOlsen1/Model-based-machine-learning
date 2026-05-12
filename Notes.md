
# PLAN 

Create individual models for predicting revenue focusing on different aspects of the data ex. actors, time, market state, etc.. 
Then we make an aggregate model that predicts a final revenue based on the predicted revenue of the indiviudal models

Ideas for individual models. 
- focus on actors 
- focus on market state
- focus on release date (day of week + month of year) 

---

## Individual Models — Feature Groups

Each model takes its feature slice → outputs a **Gaussian posterior over log(revenue)**: N(μ_i, σ_i²).
Using log-revenue keeps distributions well-behaved and roughly Gaussian.

### Model 1: Cast Popularity (actors/credits.csv)
- Features: avg past revenue of top-3 billed actors, director past revenue, cast size
- Model type: Bayesian linear regression with actor-level random effects (hierarchical)
- Latent: actor "star power" α_actor ~ N(μ_star, σ_star)
- Output: N(μ_cast, σ_cast²)
- Why: star power is strong but noisy signal — uncertainty captured via σ

### Model 2: Release Timing (release date)
- Features: month, day-of-week, holiday proximity, days-since-last-blockbuster
- Model type: Gaussian Process over calendar time OR Fourier seasonal regression
- Latent: seasonal box office cycle (yearly + weekly harmonics)
- Output: N(μ_time, σ_time²)
- Why: summer/holiday release = higher ceiling; GP captures smooth seasonality

### Model 3: Market Competition (market state)
- Features: number of wide releases same weekend, genre of competitors, total screens available
- Model type: Poisson/NegBin for competition count → linear effect on log-revenue
- Latent: market saturation factor β_comp
- Output: N(μ_market, σ_market²)
- Why: crowded weekends suppress revenue regardless of film quality

### Model 4: Genre & Keywords (movies_metadata + keywords.csv)
- Features: genre one-hot, top-50 keyword topics (LDA or TF-IDF), runtime
- Model type: Bayesian ridge regression or sparse Bayesian (ARD prior on keyword weights)
- Latent: genre-specific baseline revenue λ_genre
- Output: N(μ_genre, σ_genre²)
- Why: horror cheap-to-make → high ROI; action → high absolute revenue

### Model 5: Budget Signal
- Features: production budget (log-scaled)
- Model type: simple Bayesian linear: log(revenue) = α + β·log(budget) + ε
- Posterior: conjugate Normal-InverseGamma → closed-form N(μ_budget, σ_budget²)
- Why: budget is strongest single predictor but over-relied on alone

### Model 6: Audience Ratings (ratings.csv)
- Features: mean rating, rating count, rating variance, early-window ratings (first 7 days)
- Model type: Beta-regression on rating → revenue link via learned mapping
- Output: N(μ_rating, σ_rating²)
- Why: word-of-mouth signal; high variance rating = polarizing = niche audience

---

## Aggregate Model — Bayesian Probabilistic Fusion

### Approach: Product of Experts (Precision-Weighted Gaussian Fusion)

Each sub-model i gives posterior N(μ_i, σ_i²).  
Treat each as independent likelihood over true log-revenue z:

```
P(z | all features) ∝ ∏_i N(z | μ_i, σ_i²)
```

Combined posterior is also Gaussian (conjugate):

```
σ_agg² = 1 / Σ_i (1 / σ_i²)          # harmonic sum of precisions
μ_agg  = σ_agg² · Σ_i (μ_i / σ_i²)   # precision-weighted mean
```

**Key property**: models with high uncertainty (large σ_i) get down-weighted automatically — no manual tuning.

### Alternative: Bayesian Model Averaging (BMA)

Weight each model by marginal likelihood (how well it fits held-out data):

```
P(revenue | x) = Σ_i P(M_i | data) · P(revenue | x, M_i)
```

- Weights P(M_i | data) learned via cross-validation log-likelihood
- More expensive but selects best model family automatically

### Alternative: Stacked Bayesian Regression (recommended if models correlate)

- Run all sub-models → collect (μ_1...μ_6, σ_1...σ_6) as features
- Fit final Bayesian linear layer: log(revenue) = Σ w_i · μ_i + ε
- Put Dirichlet prior on weights w to enforce they sum to 1
- Uncertainty: σ_final² = Σ_i w_i² · σ_i² + σ_ε²

### Implementation path (Pyro)
1. Each sub-model: `pyro.sample` with feature-conditioned `dist.Normal`
2. Condition on observed log-revenue during training → SVI inference
3. Aggregator: collect `.posterior_predictive` from each → precision-weighted combine
4. Final output: `dist.Normal(μ_agg, σ_agg)` → sample for prediction intervals

### PGM Structure (high level)
```
[Cast] [Time] [Market] [Genre] [Budget] [Ratings]
  ↓       ↓      ↓       ↓       ↓        ↓
 M1      M2      M3      M4      M5       M6
  ↓       ↓      ↓       ↓       ↓        ↓
 N(μ1,σ1) ... N(μ6,σ6)
        ↓ Precision-weighted fusion ↓
              N(μ_agg, σ_agg)
                    ↓
              log(Revenue)
```