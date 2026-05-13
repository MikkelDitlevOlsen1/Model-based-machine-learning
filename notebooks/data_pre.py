import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd   # We import Pandas!
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
import itertools
from pyro.infer.autoguide import AutoNormal
import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO, Predictive
from pyro.optim import Adam, ClippedAdam

def filter_actors(data, threshold):
    # Combine all actor columns into a single series
    all_actors = pd.concat([data[f'Actor_{i}'] for i in range(1, 6)])
    
    # Count occurrences of each actor
    actor_counts = all_actors.value_counts()
    
    # Filter actors based on the threshold
    valid_actors = actor_counts[actor_counts >= threshold].index
    
    # Replace invalid actors with NaN
    for i in range(1, 6):
        data[f'Actor_{i}'] = data[f'Actor_{i}'].where(data[f'Actor_{i}'].isin(valid_actors), other=pd.NA)
    
    return data
