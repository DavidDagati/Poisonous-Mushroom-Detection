from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import warnings

def accuracy(model, X_train, X_test, y_train, y_test):

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    string = f"{model.kernel} Support Vector Machine Test Results"
    string2 = f"{model.kernel} Results Complete"

    print(f"{string:-^60}")

    print(metrics.classification_report(y_test, y_pred))

    print(f"Test set Accuracy:{metrics.accuracy_score(y_test, y_pred)}")
    print(f"Train set Accuracy {metrics.accuracy_score(y_train, y_pred_train)}")
    print(f"Training set score: {model.score(X_train, y_train)}")
    print(f"Test set score: {model.score(X_test, y_test)}")

    cm = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix\n\n", cm)

    print(f"{string2:-^60}\n\n")

def graph_svm(model, X_train, X_test, y_train, y_test):

    # # Heatmap
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(20,10)) 
    # sns.heatmap(encoded_data.corr())
    # plt.show()

    # Below is non-working svm graphing. I think this is due to the complexity of our data

    # plt.figure(figsize=(10, 8))
    # # Plotting our two-features-space
    # sns.scatterplot(x=x_train.get("cap-shape_conical"), 
    #                 y=x_train.get("cap-color_cinnamon"), 
    #                 hue=y_train, 
    #                 s=8)
    # # Constructing a hyperplane using a formula.
    # w = model.coef_[0]           # w consists of 2 elements
    # b = model.intercept_[0]      # b consists of 1 element
    # x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
    # y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
    # # Plotting a red hyperplane
    # plt.plot(x_points, y_points, c='r')
    # # Encircle support vectors
    # plt.scatter(model.support_vectors_[:, 0],
    #             model.support_vectors_[:, 1], 
    #             s=50, 
    #             facecolors='none', 
    #             edgecolors='k', 
    #             alpha=.5)
    # # Step 2 (unit-vector):
    # w_hat = model.coef_[0] / (np.sqrt(np.sum(model.coef_[0] ** 2)))
    # # Step 3 (margin):
    # margin = 1 / np.sqrt(np.sum(model.coef_[0] ** 2))
    # # Step 4 (calculate points of the margin lines):
    # decision_boundary_points = np.array(list(zip(x_points, y_points)))
    # points_of_line_above = decision_boundary_points + w_hat * margin
    # points_of_line_below = decision_boundary_points - w_hat * margin
    # # Plot margin lines
    # # Blue margin line above
    # plt.plot(points_of_line_above[:, 0], 
    #         points_of_line_above[:, 1], 
    #         'b--', 
    #         linewidth=2)
    # # Green margin line below
    # plt.plot(points_of_line_below[:, 0], 
    #         points_of_line_below[:, 1], 
    #         'g--',
    #         linewidth=2)
    # plt.show()

    return

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

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
    X = encoded_data.drop(columns=["poisonous/edible_poisonous"]) 

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=36)
    Xt = X_train.to_numpy()
    yt = y_train.to_numpy()

    # Step 3: Create the Support Vector Machine models

    linear_model = SVC(kernel='linear', random_state=2)

    # Poly can use degree to changes the outcome
    # poly_model = SVC(kernel='poly', degree= 3, random_state=8)
    # poly_model = SVC(kernel='poly', degree= 9, random_state=8)
    poly_model = SVC(kernel='poly', degree= 18, random_state=8)

    rbf_model = SVC(kernel='rbf', random_state=32)
    sigmoid_model = SVC(kernel='sigmoid', random_state=128)

    # Step 4: Predict if each mushroom is poisonous or edible

    linear_model.fit(Xt, yt)
    poly_model.fit(Xt, yt)
    rbf_model.fit(Xt, yt)
    sigmoid_model.fit(Xt, yt)

    # Step 5: Check accuracy of the model

    accuracy(linear_model, X_train, X_test, y_train, y_test)
    accuracy(poly_model, X_train, X_test, y_train, y_test)
    accuracy(rbf_model, X_train, X_test, y_train, y_test)
    accuracy(sigmoid_model, X_train, X_test, y_train, y_test)

    # Step 6: Graph