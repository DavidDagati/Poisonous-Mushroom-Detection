from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Step 1: Read Data
data = pd.read_csv("mush_data_names.csv")


# Make Data 1-hot encoded
# Keep Columns
data_encoded = pd.get_dummies(data, drop_first=False)
data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])

data_encoded.to_csv("mush_data_one_hot_encoded.csv")

# data_encoded_2 = data_encoded.drop_duplicates(keep="first")

# print(data_encoded.shape[1])

# quit()

x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 

y  = data_encoded['poisonous/edible_poisonous']


f = open('DecisionTree_Results/results_3.txt', "w")


x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=100)
#
# Create an instance of decision tree classifier
#
clf = DecisionTreeClassifier(random_state=123)
#
# Create grid parameters for hyperparameter tuning
#

params =  {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [8,10,12,18,20,30,40],
    'max_features': list(range(1,x_train.shape[1], 4)),
    'min_samples_leaf': [1,3,5,10,15,20],
    'max_depth': [3,5,7,10,15]
}
#
# Create gridsearch instance
#
grid = GridSearchCV(estimator=clf,
                    param_grid=params,
                    cv=5,
                    n_jobs=1,
                    verbose=2)
#
# Fit the model
#
grid.fit(x_train, y_train)

score_df = pd.DataFrame(grid.cv_results_)

score_df.to_csv("results_hyper.csv")

#
# Assess the score
#
print(grid.best_score_, grid.best_params_)

# print(score_df)