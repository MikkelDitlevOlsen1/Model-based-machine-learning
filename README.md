# Probabilistic Graphical Modeling: Movies Dataset Analysis

**Data Source**: [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Project Structure

```
Model-based-machine-learning/
├── data/                 # Datasets and data loading
├── notebooks/            # Jupyter notebooks for exploration and experiments
├── models/               # Pyro model implementations
├── scripts/              # Utility scripts and data processing
│   ├── __init__.py
│   ├── data_loader.py   # MovieDataLoader class
│   └── kaggle_setup.md  # Instructions for downloading dataset
├── results/              # Output, visualizations, and results
├── docs/                 # Documentation and notes
│   └── PROJECT_PLAN.md  # Current project plan and research questions
└── README.md
```

## Dataset Overview

**The Movies Dataset** contains:
- **45,000+ movies** with metadata (budget, revenue, genres, runtime, ratings)
- **26M+ user ratings** (or 100K+ in small version for quick testing)
- **Cast and crew** information for each movie
- **Keywords/tags** associated with movies

### Key Files
- `movies_metadata.csv` - Movie attributes
- `ratings.csv` / `ratings_small.csv` - User ratings
- `credits.csv` - Cast and crew
- `keywords.csv` - Movie keywords

## Requirements

- Python 3.8+
- Pyro
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Kaggle CLI (for data download)

## Project Phases

1. **Data Exploration** - Understand dataset structure, distributions, relationships
2. **Research Question** - Define specific modeling problem
3. **PGM Formulation** - Design probabilistic model structure
4. **Implementation** - Code model in Pyro
5. **Inference** - Train model and perform inference
6. **Evaluation** - Validate model and interpret results
