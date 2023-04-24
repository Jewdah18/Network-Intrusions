import pandas as pd

df = pd.read_csv(r'C:\Users\jdrel\OneDrive\Documents\Data_Science\Springboard\Capstone-2\data\processed\Full KDD Data.csv')
df_y = pd.concat([df, data['labels']], axis = 1)
for label_i in df_y['labels'].unique():
    for label_j in df_y['labels'].unique():
        # Take only the data that has smurf or neptune
        blue_data = df_y.loc[df_y['labels'].isin([label_i,label_j])]
        # Grab only the features
        x_blue = blue_data.drop('labels', axis = 1)
        # create y that is 0 for smurf and 1 for neptune
        y_diff_attacks = np.where(blue_data['labels'] == 'label_i', 0, 1)
        # Create Lasso model
        lasso = Lasso(max_iter = 50000)

        # Define hyperparameter grid with a value less than 00.5 since that was the be
        params = {'alpha': np.linspace(.001, 5, 20)}

        # Perform grid search
        grid_search = GridSearchCV(estimator=lasso, param_grid=params, cv=8)

        # fit the gridsearch of parameters to the data
        grid_search.fit(x_blue, y_diff_attacks)

        # Print best hyperparameters
        print("Best hyperparameters: ", grid_search.best_params_)
        
        # Find the coefficients of lasso regularization
        lasso = Lasso(alpha = .001)
        # Fit the lasso regularization to the data 
        lasso.fit(x_blue, y_diff_attacks)
        # Create a dictionary of all the features and their corresponding lasso coefficients
        lasso_dict = {df.columns[i]:lasso.coef_[i] for i in range(len(df.columns)) if list(lasso.coef_)[i] != 0}
        # Create a list of all the features that don't have a lasso coefficient of zero
        lasso_features = [df.columns[i] for i in range(len(df.columns)) if list(lasso.coef_)[i] != 0]
        # Print the features and coefficients
        print(lasso_dict)
        # Print the number of features are left from lasso
        print(len(lasso_dict))