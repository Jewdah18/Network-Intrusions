import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from tqdm import tqdm

df_y = pd.read_csv('./data/processed/KDD Data.csv')
df = df_y.drop(['Unnamed: 0', 'labels'], axis = 1)

cats = ['service', 'flag', 'protocol_type']

cats_cols = []

for prefix in cats:
    cols = df.filter(regex=prefix).columns
    df[cols] = df[cols].astype('category')
    cols = list(df.columns.get_indexer(cols))
    cats_cols.extend(cols)
    
# Initialize cost list
cost = []

# Iterate for different amounts of centroids
for k in tqdm(range(6,11)):
    # Use K-Prototype to generate a model. These parameters can change but hopefully we find something good
    KProto = KPrototypes(n_clusters = k, init = 'Cao', n_init = 10, 
                         gamma = 0.125,  verbose = 0, n_jobs = -1)
    # Use the KPrototypes model on the actual data
    cluster_labels = KProto.fit_predict(df, categorical = cats_cols)
    # Append cost to the cost list for graphing later
    cost.append(KProto.cost_) 

# An elbow plot of k vs the cost function to try to find the optimal k
plt.plot(range(6,11), cost, marker = 'o');
# Label the x-axis
plt.xlabel('Number of Clusters');
#label the y-axis
plt.ylabel('Dissimilarity');
#Give the plot a title
plt.title('Elbow Plot for KPrototypes');