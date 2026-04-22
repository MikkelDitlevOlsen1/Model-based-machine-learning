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

## Getting Started

### 1. Set Up Data
Follow [scripts/kaggle_setup.md](scripts/kaggle_setup.md) to download The Movies Dataset:
```bash
# Install Kaggle CLI
pip install kaggle

# Configure credentials and download
kaggle datasets download -d rounakbanik/the-movies-dataset
unzip the-movies-dataset.zip -d data/
```

### 2. Load Data
```python
from scripts.data_loader import MovieDataLoader

loader = MovieDataLoader(data_path='data/')
loader.load_all(use_small_ratings=True)  # Use small dataset initially
loader.explore_movies()
loader.explore_ratings()
```

### 3. Choose Research Question
Edit [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) and select one of the research questions:
- Rating prediction model
- Revenue modeling
- User preference inference
- Temporal trend analysis
- Joint latent factor model

### 4. Implement PGM in Pyro
Create model files in `models/` and notebooks in `notebooks/`

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
