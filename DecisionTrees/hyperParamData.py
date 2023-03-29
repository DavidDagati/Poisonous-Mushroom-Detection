from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_curve, auc

import json
import time


# results = pd.read_csv('results_hyper_final.csv')

# testParam = []

# for row in results.iterrows():
#     if(int(row[-1]['rank_test_score']) == 1):
#         testParam.append(eval(row[-1]['params']))


# # print(testParam)


data = pd.read_csv("mush_data_names.csv")


# Make Data 1-hot encoded
# Keep Columns
data_encoded = pd.get_dummies(data, drop_first=False)
data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])

# data_encoded.to_csv("mush_data_one_hot_encoded.csv")

# data_encoded_2 = data_encoded.drop_duplicates(keep="first")


x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 

y  = data_encoded['poisonous/edible_poisonous']

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=42)

# left = len(testParam)

# finalParams = []

# for params in testParam:
#     # print(params['max_depth'])
#     maxFalse = 0
#     print(f"There is {left} left")
   
#     # f.write(f"\nCriterion={params['criterion']} | Splitter={splitter} | maxDepth={maxDepth} | minSampleSplit={minSampleSplit} | maxFeatures={maxFeatures}\n")
#     clf = None
#     y_pred = None
#     y_pred_train = None
#     cm = None
#     for r in [0, 3, 8, 15, 27, 31, 48, 51, 63, 75, 88]:
#         clf = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'], random_state=r, min_samples_split=params['min_samples_split'], max_features=params['max_features'])

#         startTime = time.time()
#         clf.fit(x_train,y_train)
#         enclfime = time.time()

#         y_pred = clf.predict(x_test)

#         y_pred_train = clf.predict(x_train)

#         cm = confusion_matrix(y_test, y_pred)

#         maxFalse += cm[1][0] + cm[0][1]

#     if(maxFalse == 0):
#         finalParams.append(params)
#         f.write("-"*50)
#         f.write("\nClassification Report:")
#         f.write(classification_report(y_test, y_pred) + "\n\n")



#         # fig = plt.figure(figsize=(25,20))
#         # _ = tree.plot_tree(clf, 
#         #                     feature_names=data_encoded.columns.values,  
#         #                     class_names="pe",
#         #                     label="all",
#         #                     filled=True)

#         # fig.savefig(f"DecisionTree_Results/decistion_tree_FINAL.png")


#         # Compare train-set and test-set accuracy
#         # https://gist.github.com/SaranyaRavikumar06/f8a76c500f954fdfd927c74849bd24c3

#         f.write(f'Runtime: {enclfime - startTime}\n')

#         f.write(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}\n")

#         f.write(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}\n")

#         f.write(f'Training set score: {clf.score(x_train, y_train)}\n')

#         f.write(f'Test set score: {clf.score(x_test, y_test)}\n')



#         f.write(f'\nConfusion matrix\n\n {cm}\n')

#         f.write("-"*50)
#     left -= 1



# print(len(finalParams)) # 1151

# print(finalParams[0])

# res = {
#     'criterion': {},
#     'max_depth': {}, 
#     'max_features': {}, 
#     'min_samples_leaf': {},
#     'min_samples_split': {}
# }

# for param in finalParams:
#     res['criterion'][param['criterion']] = res['criterion'].get(param['criterion'], 0) + 1
#     res['max_depth'][param['max_depth']] = res['max_depth'].get(param['max_depth'], 0) + 1
#     res['max_features'][param['max_features']] = res['max_features'].get(param['max_features'], 0) + 1
#     res['min_samples_leaf'][param['min_samples_leaf']] = res['min_samples_leaf'].get(param['min_samples_leaf'], 0) + 1
#     res['min_samples_split'][param['min_samples_split']] = res['min_samples_split'].get(param['min_samples_split'], 0) + 1


# out_file.close()
# out_file = open("DT_Data/paramCounts.json", "w")

# print(len())

# json.dump(res, out_file)

# # f.close()

out_file = open("DT_Data/finalParams.json", "r")


# json.dump(finalParams, out_file)

finalParams = json.load(out_file)

params = None

f = open('DecisionTree_Results/results_Final_HyperParams_3.txt', "w")

for m in finalParams:
    if(m['criterion'] == 'entropy' and m['max_depth'] == 10 and m['max_features'] == 41 and m['min_samples_leaf'] == 3 and m['min_samples_split'] == 12):
        params = m

clf = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'], random_state=42, min_samples_split=params['min_samples_split'], max_features=params['max_features'])

startTime = time.time()
clf.fit(x_train,y_train)
enclfime = time.time()

y_pred = clf.predict(x_test)

y_pred_train = clf.predict(x_train)

cm = confusion_matrix(y_test, y_pred)


f.write("-"*50)
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