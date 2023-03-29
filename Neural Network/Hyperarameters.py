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

## Hyperparameters 
nn = MLPClassifier(max_iter=300, random_state=1)
parameter_space = {
    'hidden_layer_sizes': [(100,50,30), (20,), (5,2)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=5)
clf.fit(x, y)

print('Best parameters found:\n', clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , clf.predict(x_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred))

endTime = time.time()
print("Time: ", endTime - startTime)

