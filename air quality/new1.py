import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
train = pd.read_csv('Train.csv')
X_train = train.drop(columns=['air_quality_index'], axis=1)
y_train = train['air_quality_index']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define functions for predicting and classifying air quality
def classify_air_quality(value):
    if value > 0:
        return "Good"
    else:
        return "Bad"

def predict_air_quality(features):
    prediction = model.predict([features])
    air_quality = classify_air_quality(prediction[0])
    return air_quality

# Define function for button click event
def predict():
    try:
        features = [float(entry.get()) for entry in feature_entries]
        air_quality_prediction = predict_air_quality(features)
        result_label.config(text=f"Predicted Air Quality: {air_quality_prediction}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all features.")

# Create Tkinter window
window = tk.Tk()
window.title("Air Quality Predictor")

# Create labels and entry fields for features
feature_labels = ["NO2:", "SO2:", "Ozone:", "CO:", "Humidity:"]
feature_entries = []
for i, label_text in enumerate(feature_labels):
    label = ttk.Label(window, text=label_text)
    label.grid(row=i, column=0, sticky="e", padx=10, pady=5)
    entry = ttk.Entry(window)
    entry.grid(row=i, column=1, padx=10, pady=5)
    feature_entries.append(entry)

# Create predict button
predict_button = ttk.Button(window, text="Predict", command=predict)
predict_button.grid(row=len(feature_labels), columnspan=2, pady=10)

# Create label to display result
result_label = ttk.Label(window, text="")
result_label.grid(row=len(feature_labels) + 1, columnspan=2, pady=5)

# Run the Tkinter event loop
window.mainloop()
