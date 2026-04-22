# Setting Up Kaggle Dataset

## Prerequisites
```bash
pip install kaggle pandas numpy
```

## Download Instructions

### Option 1: Using Kaggle CLI (Recommended)
1. Create Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<YourUsername>\.kaggle\`)

2. Set permissions (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download dataset:
   ```bash
   kaggle datasets download -d rounakbanik/the-movies-dataset
   unzip the-movies-dataset.zip -d ../data/
   ```

### Option 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
2. Click "Download" button
3. Extract to `data/` folder

## Files in Dataset
- `movies_metadata.csv` (~350 MB) - 45,000 movies
- `ratings.csv` (~1.1 GB) - 26M ratings
- `ratings_small.csv` (~200 MB) - 100K ratings (for quick testing)
- `credits.csv` (~170 MB) - Cast/crew info
- `keywords.csv` (~3 MB) - Keywords/tags

## Quick Start
```python
import pandas as pd

# Load small dataset for exploration
movies = pd.read_csv('../data/movies_metadata.csv')
ratings = pd.read_csv('../data/ratings_small.csv')
credits = pd.read_csv('../data/credits.csv')

print(movies.shape)
print(ratings.shape)
print(credits.shape)
```
