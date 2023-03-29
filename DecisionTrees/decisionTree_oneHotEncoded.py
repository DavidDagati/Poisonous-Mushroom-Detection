import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_curve, auc

import time

'''
DATASET:
Preprocessing => One-hot with all Columns

WHAT TO TEST...
https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680



---

criterion: (default = “gini") the metric used to create splits. Use “entropy" for information gain.
max_depth: (default = None) the maximum number of layers your tree will have. When None, the layers will continue until a pure split is achieved or another min/max parameter is achieved.
min_samples_split: (default = 2) the minimum number of samples in an internal node that allows for a split to occur. If a node has less than this number if becomes a leaf (terminal node).
min_samples_leaf: (default = 1) the minimum number of samples required for a leaf node. A split will only occur if the nodes that result from the split meet this minimum. This can be especially useful in regression trees.


FOR TESTING MODEL:
1. Training Accuracy Score
2. Validation Accuracy Score

'''




if __name__ == "__main__":
    # Step 1: Read Data
    data = pd.read_csv("mush_data_names.csv")


    # Make Data 1-hot encoded
    # Keep Columns
    data_encoded = pd.get_dummies(data, drop_first=False)
    data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])
    
    # data_encoded.to_csv("mush_data_one_hot_encoded.csv")

    # data_encoded_2 = data_encoded.drop_duplicates(keep="first")


    x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 

    y  = data_encoded['poisonous/edible_poisonous']


    f = open('DecisionTree_Results/results_2.txt', "w")

    '''
    Criterion => Entropy or gini

    splitter => best vs random

    max_depth => None, 3,5,7

    min_samples_leaf 
    -> avoid overfitting
    -> ensures that each leaf has more than one element

    max_features => 10, 15, None
    -> num of features to consider for the best split
    '''


    # for crit in ['entropy', 'gini']:
    #     for s in ['best', 'random']:
    #         runDecisionTree(crit, s, 5, 2, None,1, x, y, data_encoded, f)
    
    # plt.clf()


    # Best is 7
    # testMaxDepth(crit, s, 1, None, x, y, data_encoded, f)

    # plt.clf()

    # # Best is 62
    # testMaxFeatures(crit, s, 1, 7, x, y, data_encoded, f)

    # plt.clf()


    # # Best is 0.0
    # testMinSampleLeaves(crit, s, x, y)

    # plt.clf()



    # # Best is 0.01
    # testSampleSplit(crit, s, x, y)

    # plt.clf()

    f.close()
    f = open('DecisionTree_Results/results_Final_HyperParams.txt', "w")

    params = {'criterion': 'gini', 'max_depth': 7, 'max_features': 25, 'min_samples_leaf': 1, 'min_samples_split': 8}




    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=100)


    f.write("-"*50)

    # f.write(f"\nCriterion={params['criterion']} | Splitter={splitter} | maxDepth={maxDepth} | minSampleSplit={minSampleSplit} | maxFeatures={maxFeatures}\n")


    clf = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'], random_state=100, min_samples_split=params['min_samples_split'], max_features=params['max_features'])

    startTime = time.time()
    clf.fit(x_train,y_train)
    enclfime = time.time()

    y_pred = clf.predict(x_test)

    y_pred_train = clf.predict(x_train)


    f.write("\nClassification Report:")
    f.write(classification_report(y_test, y_pred) + "\n\n")



    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                       feature_names=data_encoded.columns.values,  
                       class_names="pe",
                       label="all",
                       filled=True)

    fig.savefig(f"DecisionTree_Results/decistion_tree_FINAL.png")


    # Compare train-set and test-set accuracy
    # https://gist.github.com/SaranyaRavikumar06/f8a76c500f954fdfd927c74849bd24c3

    f.write(f'Runtime: {enclfime - startTime}\n')

    f.write(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}\n")

    f.write(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}\n")

    f.write(f'Training set score: {clf.score(x_train, y_train)}\n')

    f.write(f'Test set score: {clf.score(x_test, y_test)}\n')


    cm = confusion_matrix(y_test, y_pred)

    f.write(f'\nConfusion matrix\n\n {cm}\n')

    f.write("-"*50)



    f.close()