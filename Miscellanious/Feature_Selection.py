import numpy as np
import pandas as pd
import os 
from sklearn.linear_model import Lasso
from tqdm import tqdm

# import data to the script
df_y = pd.read_csv(r'C:\Users\jdrel\OneDrive\Documents\Data_Science\Springboard\Capstone-2\data\processed\Full KDD Data.csv')
# drop labels to have only features
df = df_y.drop(['Unnamed: 0','labels'], axis = 1)
relevent_features = []

relevent_features = []
# Iterate over all the types of intrusions 
for i, label_i in tqdm(enumerate(df_y['labels'].unique())):
    # Iterate over a different intrusion
    for j, label_j in tqdm(enumerate(df_y['labels'].unique())):
        if i < j:
            # Take only the data that has smurf or neptune
            blue_data = df_y.loc[df_y['labels'].isin([label_i, label_j])]
            # Grab only the features
            x_blue = blue_data.drop('labels', axis = 1)
            # create y that is 0 for smurf and 1 for neptune
            y_diff_attacks = np.where(blue_data['labels'] == label_i, 0, 1)

            # Lasso would not converge and it would take too long to use gridsearch
            # most of the time alpha values were from .001 to .002
            lasso = Lasso(alpha = .002, max_iter = 50000)

            # Fit the lasso regularization to the data 
            lasso.fit(x_blue, y_diff_attacks)

            # Create a dictionary of all the features and their corresponding lasso coefficients
            lasso_list = [df.columns[i] for i in range(len(df.columns)) if list(lasso.coef_)[i] != 0]

            relevent_features.extend(lasso_list)

print(relevent_features)
