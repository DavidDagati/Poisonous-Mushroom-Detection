from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA


# Step 0: Prepare the dataframes

# original_data = pd.read_csv("mush_data_names.csv")
# print(original_data.head())

# Step 1: Encode the Data

# encoded_data = pd.get_dummies(original_data, drop_first=False)
# encoded_data = encoded_data.drop(columns=["poisonous/edible_edible"])

# encoded_data = pd.get_dummies(data, drop_first=True)
# encoded_data.to_csv("mush_data_one_hot_encoded.csv")

encoded_data = pd.read_csv("mush_data_one_hot_encoded.csv")

# Step 2: Separate data into an x (All data except the poisonous/edible column) and y (Only the poisonous/edible column)

y = encoded_data['poisonous/edible_poisonous']
x = encoded_data.drop(columns=["poisonous/edible_poisonous"]) 

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=36)

# Step 3: Create the Support Vector Machine model

model = SVC(gamma='auto')


# Step 4: Predict if each mushroom is poisonous or edible

model.fit(x_train, y_train)

# Step 5: Check accuracy of the model

# print(f'Accuracy: {model.score(x,y):.3f}')

y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)

print(metrics.classification_report(y_test, y_pred))

print(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}")
print(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}")
print(f'Training set score: {model.score(x_train, y_train)}' )
print(f'Test set score: {model.score(x_test, y_test)}')

cm = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

# Step 6: Graph

# TODO get it working

pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(x_train)


# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X_train2, 
                      y=y_train.values,
                      clf=model, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(x.columns[0], size=14)
plt.ylabel(x.columns[1], size=14)
plt.title('Poisonous and Edible Mushrooms: Support Vector Machine', size=16)