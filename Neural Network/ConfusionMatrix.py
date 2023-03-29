import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


startTime = time.time()
data = pd.read_csv("mush_data_names.csv")
data_encoded = pd.get_dummies(data, drop_first=False)
data_encoded = data_encoded.drop(columns=["poisonous/edible_edible"])
x = data_encoded.drop(columns=["poisonous/edible_poisonous"]) 
y  = data_encoded['poisonous/edible_poisonous']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


## Graph
nnn = MLPClassifier(activation='logistic', hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs', alpha=0.0001, learning_rate= 'adaptive', max_iter=300)
nnn.fit(x_train, y_train)
testPred=nnn.predict(x_test)
print("(activation='logistic', hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs', alpha=0.0001, learning_rate= 'adaptive', max_iter=300)")
print('Accuracy: ', nnn.score(x_test, y_test.values))

fig = plot_confusion_matrix(nnn, x_test, y_test, display_labels=nnn.classes_)
fig.figure_.suptitle("Confusion Matrix")
plt.show()

endTime = time.time()
print("Time: ", endTime - startTime)
