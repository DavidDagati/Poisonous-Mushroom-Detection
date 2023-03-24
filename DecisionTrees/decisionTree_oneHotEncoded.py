import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import OneHotEncoder

'''
DATASET:
Preprocessing => One-hot with all Columns

WHAT TO TEST...
https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680


Criterion => Entropy or gini

splitter => best vs random

max_depth => None, 3,5,7

min_samples_leaf 
-> avoid overfitting
-> ensures that each leaf has more than one element

max_features => 10, 15, None
-> num of features to consider for the best split

---

criterion: (default = “gini”) the metric used to create splits. Use “entropy” for information gain.
max_depth: (default = None) the maximum number of layers your tree will have. When None, the layers will continue until a pure split is achieved or another min/max parameter is achieved.
min_samples_split: (default = 2) the minimum number of samples in an internal node that allows for a split to occur. If a node has less than this number if becomes a leaf (terminal node).
min_samples_leaf: (default = 1) the minimum number of samples required for a leaf node. A split will only occur if the nodes that result from the split meet this minimum. This can be especially useful in regression trees.


FOR TESTING MODEL:
1. Training Accuracy Score
2. Validation Accuracy Score

'''

# Step 1: Read Data
data = pd.read_csv("mush_data_names.csv")

# Make Data 1-hot encoded
# Keep Columns
data_encoded = pd.get_dummies(data, drop_first=False)
data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])

# 1-hot encoded -> remove columns
data_encoded = pd.get_dummies(data, drop_first=True)


data_encoded.to_csv("mush_data_one_hot_encoded.csv")


x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 

y  = data_encoded['poisonous/edible_poisonous']

# print(data_encoded)

# print(x)

# print(y)

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.5, random_state=42)


clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=100, min_samples_leaf=5)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

y_pred_train = clf.predict(x_train)

# print("Validation Score:",metrics.score())


# print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf, 
#                    feature_names=data_encoded.columns.values,  
#                    class_names="pe",
#                    label="all",
#                    filled=True)

# fig.savefig("decistion_tree.png")


# Compare train-set and test-set accuracy
# https://gist.github.com/SaranyaRavikumar06/f8a76c500f954fdfd927c74849bd24c3


print(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}")

print(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}")

print(f'Training set score: {clf.score(x_train, y_train)}' )

print(f'Test set score: {clf.score(x_test, y_test)}')


cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)