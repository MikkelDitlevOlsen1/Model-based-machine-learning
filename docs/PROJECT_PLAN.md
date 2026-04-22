# Project Plan: Movies Dataset Analysis

## Approach: Data-Driven
- [x] Problem-driven
- [x] Data-driven
- [ ] Model-driven
- [ ] Paper-driven

## Data Source
**Kaggle: The Movies Dataset**
- URL: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- Files:
  - `movies_metadata.csv` - Movie information (title, budget, revenue, genres, release date, etc.)
  - `ratings.csv` - User ratings (userId, movieId, rating, timestamp)
  - `credits.csv` - Cast and crew information
  - `keywords.csv` - Movie keywords/tags

## Research Question
[TO BE DEFINED - Choose from examples below]

### Possible Research Questions:
1. How do genres, budget, and cast influence movie ratings?
2. What factors predict commercial success (revenue vs budget)?
3. Can we model user preferences and predict ratings using latent factors?
4. How do temporal trends affect movie ratings and revenue?
5. Joint model of rating prediction and revenue estimation?

## Probabilistic Graphical Model
[TO BE DEFINED - After choosing research question]

### Model Structure (Example):
- Observed: Ratings, Genres, Budget, Cast
- Latent: User preferences, Movie quality factors
- Directed edges: Connect factors → observed variables

## Implementation Plan
- [ ] Download and explore dataset
- [ ] Clean and preprocess data
- [ ] Define specific research question
- [ ] Formalize PGM structure
- [ ] Implement in Pyro
- [ ] Train and evaluate model
- [ ] Analyze results and insights

## Results & Findings
[TO BE UPDATED - Document your results]
