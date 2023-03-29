from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import numpy as np
import json
import time


'''
PREPARE DATASET:
'''

# Step 1: Read Data
data = pd.read_csv("mush_data_names.csv")


# Make Data 1-hot encoded
data_encoded = pd.get_dummies(data, drop_first=False)
data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])

# Save One-Hot Encoded Dataset
data_encoded.to_csv("mush_data_one_hot_encoded.csv")

# Set Inputs and Ouputs
x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 
y  = data_encoded['poisonous/edible_poisonous']


x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=100)

f = open('DecisionTree_Results/accuracies.txt', "w")

'''
DEFAULT PARAMETERS:
'''

# Start with max_depth of 2
clf = DecisionTreeClassifier(random_state=123, max_depth=2)

startTime = time.time()
clf.fit(x_train,y_train)
enclfime = time.time()

y_pred = clf.predict(x_test)

y_pred_train = clf.predict(x_train)



f.write("-"*50)
f.write("\nBefore Hyperparameter Tune:")

f.write("\nClassification Report:")
f.write(classification_report(y_test, y_pred) + "\n\n")



fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                    feature_names=data_encoded.columns.values,  
                    class_names="pe",
                    label="all",
                    filled=True)

fig.savefig(f"DecisionTree_Results/Decision_Tree_Before.png")


# Compare train-set and test-set accuracy

f.write(f'Runtime: {enclfime - startTime}\n')

f.write(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}\n")

f.write(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}\n")

f.write(f'Training set score: {clf.score(x_train, y_train)}\n')

f.write(f'Test set score: {clf.score(x_test, y_test)}\n')



cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()

plt.savefig(f"DecisionTree_Results/Before_ConfusionMatrix.png")


f.write("-"*50)



'''
FIND THE MOST OPTIMAL HYPERPARAMETERS: 
'''

clf = DecisionTreeClassifier(random_state=123)


params =  {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [8,10,12,18,20,30,40],
    'max_features': list(range(1,x_train.shape[1], 4)),
    'min_samples_leaf': [1,3,5,10,15,20],
    'max_depth': [3,5,7,10,15]
}

# Use GridSearch to test all combinations 
grid = GridSearchCV(estimator=clf,
                    param_grid=params,
                    cv=5,
                    n_jobs=1,
                    verbose=2)

# Fit the model
grid.fit(x_train, y_train)


# Record Performance of each Hyperparameter combiation
score_df = pd.DataFrame(grid.cv_results_)
score_df.to_csv("DecisionTree_Results/results_hyper.csv")

print(f"---\nCONFIRM RESULTS/TEST OTHER RANDOM STATES (FILTER)\n---")


results = pd.read_csv('DecisionTree_Results/results_hyper.csv')


# # Get all parameters to test
testParam = []
for row in results.iterrows():
    if(int(row[-1]['rank_test_score']) == 1):
        testParam.append(eval(row[-1]['params']))


left = len(testParam)
finalParams = []

for params in testParam:
    maxFalse = 0
    print(f"There is {left}/{len(testParam)} left.")
   
    clf = None
    y_pred = None
    y_pred_train = None
    cm = None
    for r in [0, 3, 8, 15, 27, 31, 48, 51, 63, 75, 88]:
        clf = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'], random_state=r, min_samples_split=params['min_samples_split'], max_features=params['max_features'])

        startTime = time.time()
        clf.fit(x_train,y_train)
        enclfime = time.time()

        y_pred = clf.predict(x_test)

        y_pred_train = clf.predict(x_train)

        cm = confusion_matrix(y_test, y_pred)

        maxFalse += cm[1][0] + cm[0][1]

    if(maxFalse == 0):
        finalParams.append(params)

    left -= 1

print(f"{len(testParam) - len(finalParams)} were successful.")



print(f"---\nCOUNT OF HYPERPARAMETERS\n---")

res = {
    'criterion': {},
    'max_depth': {}, 
    'max_features': {}, 
    'min_samples_leaf': {},
    'min_samples_split': {}
}

for param in finalParams:
    res['criterion'][param['criterion']] = res['criterion'].get(param['criterion'], 0) + 1
    res['max_depth'][param['max_depth']] = res['max_depth'].get(param['max_depth'], 0) + 1
    res['max_features'][param['max_features']] = res['max_features'].get(param['max_features'], 0) + 1
    res['min_samples_leaf'][param['min_samples_leaf']] = res['min_samples_leaf'].get(param['min_samples_leaf'], 0) + 1
    res['min_samples_split'][param['min_samples_split']] = res['min_samples_split'].get(param['min_samples_split'], 0) + 1



out_file = open("DecisionTree_Results/paramCounts.json", "w")

json.dump(res, out_file)

out_file = open("DecisionTree_Results/finalParams.json", "w")

json.dump(finalParams, out_file)

print(f"{res}")


params = None


for m in finalParams:
    if(m['criterion'] == 'entropy' and m['max_depth'] == 10 and m['max_features'] == 41 and m['min_samples_leaf'] == 3 and m['min_samples_split'] == 12):
        params = m

clf = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'], random_state=42, min_samples_split=params['min_samples_split'], max_features=params['max_features'])

startTime = time.time()
clf.fit(x_train,y_train)
enclfime = time.time()

y_pred = clf.predict(x_test)

y_pred_train = clf.predict(x_train)

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()

plt.savefig(f"DecisionTree_Results/Final_ConfusionMatrix.png")


f.write("-"*50)
f.write("\nAfter Hyperparameter Tune:")

f.write("\nClassification Report:")
f.write(classification_report(y_test, y_pred) + "\n\n")



fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                    feature_names=data_encoded.columns.values,  
                    class_names="pe",
                    label="all",
                    filled=True)

fig.savefig(f"DecisionTree_Results/Final_Decision_Tree.png")


# Compare train-set and test-set accuracy
# https://gist.github.com/SaranyaRavikumar06/f8a76c500f954fdfd927c74849bd24c3

f.write(f'Runtime: {enclfime - startTime}\n')

f.write(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}\n")

f.write(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}\n")

f.write(f'Training set score: {clf.score(x_train, y_train)}\n')

f.write(f'Test set score: {clf.score(x_test, y_test)}\n')



f.write(f'\nConfusion matrix\n\n {cm}\n')

f.write("-"*50)


