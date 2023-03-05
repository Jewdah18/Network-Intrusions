# Import the Data Science Essentials
import pandas as pd
import numpy as np

from kmodes.kprototypes import KPrototypes
from gap_statistic import OptimalK

# Import data and variables from EDA notebook
X = pd.read_csv('./data/interim/X_scaled.csv')
# The column index is 34-40 as it was appended on the other columns
cat_col_index = [i for i in range(34,41)]

# Run the gap statistic index calculation.
optimalK = OptimalK(parallel_backend = 'rust')


