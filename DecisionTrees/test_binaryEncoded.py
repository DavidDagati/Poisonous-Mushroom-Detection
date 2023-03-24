import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import category_encoders as ce

# Step 1: Read Data
data = pd.read_csv("mush_data_names.csv")

# print(data.columns)

data_cols = list(data.columns)

# print(cols)

# # Step 2: Use Binary Encoding 
encoder = ce.BinaryEncoder(cols=data_cols,return_df=True)

# print(encoder.get_feature_names_out)


data_encoded=encoder.fit_transform(data) 

print(data_encoded.head())

# # data_encoded.to_csv("mush_data_binary_encoded.csv")


# x = data_encoded.drop(columns=["poisonous/edible_1"]) 

# y  = data_encoded['poisonous/edible_1']

# print(x)

# print(y)

# x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.3, random_state=100)


# clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=4, min_samples_leaf=5)

# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# # # print(confusion_matrix(y_test, y_pred))
# # # print(classification_report(y_test, y_pred))



# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf, 
#                    feature_names=data_encoded.columns.values,  
#                    class_names="pe",
#                    label="all",
#                    filled=True)

# fig.savefig("decistion_tree.png")