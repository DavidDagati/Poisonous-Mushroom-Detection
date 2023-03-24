import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("mush_data_numbers.csv")
y = data["poisonous/edible"]
x = data.drop("poisonous/edible", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

nn = MLPClassifier(activation='logistic', hidden_layer_sizes=(30, 25), random_state=1, solver='sgd')

nn.fit(x_train, y_train)

pred=nn.predict(x_test)
a=y_test.values

count = 0
for i in range(len(pred)):
    if pred[i] == a[i]:
        count = count + 1

print((count/len(pred))*100)
    
