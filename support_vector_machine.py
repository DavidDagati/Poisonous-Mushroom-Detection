from sklearn.svm import SVC
import pandas as pd

# Step 0: Prepare the dataframes
original_data = pd.read_csv("mush_data_numbers.csv")
# print(original_data.head())

# Step 1: Seperate data into an x (All data except the poisonous/edible column) and y (Only the poisonous/edible column)

y = original_data["poisonous/edible"]
x = original_data.drop("poisonous/edible", axis = 1)

# Step 2: Create the Support Vector Machine model

model = SVC(gamma='auto')

# Step 3: Predict if each Mmshroom is posionous or edible

model.fit(x,y)

# Step 4: Check accuracy of the model

print(f'Accuracy: {model.score(x,y):.3f}')