import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("dataset/house_data.csv")

X = data[['Area', 'Rooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model/house_model.pkl", "wb"))
print("Model trained and saved successfully")