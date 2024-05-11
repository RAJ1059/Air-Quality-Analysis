import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
X_train = train.drop(columns=['air_quality_index'], axis = 1)
y_train = train['air_quality_index']

model=LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(test)

def classify_air_quality(value):
    if value > 0:
        return "Good"
    else:
        return "Bad"

def predict_air_quality(features):
    prediction = model.predict([features])
    air_quality = classify_air_quality(prediction[0])
    return air_quality

sample_features = [-0.5, -0.3, -0.2, -0.4, -1.5]  # Example set of features
air_quality_prediction = predict_air_quality(sample_features)
print("Predicted Air Quality:", air_quality_prediction)