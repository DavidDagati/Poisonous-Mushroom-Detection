from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

if __name__ == "__main__":
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
    xt = x_train.to_numpy()
    yt = y_train.to_numpy()

    # Step 3: Create the Support Vector Machine model

    model = SVC(gamma='auto')


    # Step 4: Predict if each mushroom is poisonous or edible

    # model.fit(x_train, y_train)
    model.fit(xt, yt)


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

    # Heatmap
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(20,10)) 
    # sns.heatmap(encoded_data.corr())
    # plt.show()

    pca = PCA(n_components=95)
    scaler = StandardScaler()
    X_train = pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    plt.figure(figsize=(8,6))
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='plasma')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.show()