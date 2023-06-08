import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# set random seed for reproducibility
np.random.seed(123)

# create categorical columns
cat_col1 = pd.Series(np.random.choice(['A', 'B', 'C'], size=1000))
cat_col2 = pd.Series(np.random.choice(['X', 'Y', 'Z'], size=1000))

# create numerical columns
num_cols = pd.DataFrame(np.random.randn(1000, 8), columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7', 'Num8'])

# concatenate categorical and numerical columns
df = pd.concat([cat_col1, cat_col2, num_cols], axis=1)

# set column names for categorical columns
df.columns = ['Cat1', 'Cat2', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7', 'Num8']

# Perform the clustering for the dataset
KProto = KPrototypes(n_clusters = 8, init = 'cao', n_init = 10, verbose = 0)
# Use the KPrototypes model on the actual data
labels = KProto.fit_predict(df, categorical = [0,1])


# Create Bootstrap dataset with the same index as before so that I can find the original clusters as well
bootstrap = df.sample(frac = 0.2, replace = True)

# Re run the clustering for the sample
KProto = KPrototypes(n_clusters = 8, init = 'random', n_init = 10, verbose = 0)
# Use the KPrototypes model on the actual data
bootstrap_labels = KProto.fit_predict(bootstrap, categorical = [0,1])

# Create Original Dataset's lables for bootstrapped data
orig_value = [labels[i] for i in bootstrap.index]

# Rand index calculation
Rand = adjusted_rand_score(orig_value, bootstrap_labels)

# Normalized Mutual index calculation
NMi = normalized_mutual_info_score(orig_value, bootstrap_labels)

# Fowlkes Mallow Index calculation
FMi = fowlkes_mallows_score(orig_value, bootstrap_labels)

print(f"Adjusted Rand Index: {Rand}")
print(f"Normalized Mutual Index: {NMi}")
print(f"Fowlkes Mallow Index: {FMi}")