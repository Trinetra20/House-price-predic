import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt


data = pd.read_csv('housing.csv')


non_categorical_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
data_categorical = data.drop(columns=non_categorical_columns)
data_categorical_encoded = pd.get_dummies(data_categorical, drop_first=True)
data_encoded = pd.concat([data[non_categorical_columns], data_categorical_encoded], axis=1)


X = data_encoded.drop('price', axis=1)
y = data_encoded['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


root = tk.Tk()
root.title("House Price Predictor")


def predict_price():
    try:
       
        inputs = [entry.get().lower() == 'yes' for entry in entries]
        scaled_inputs = scaler.transform([inputs])
       
        predicted_price = model.predict(scaled_inputs)[0]
     
        messagebox.showinfo("Predicted Price", f"The predicted price is ${predicted_price:.2f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid inputs (yes/no).")

entries = []
for i, feature in enumerate(X.columns):
    tk.Label(root, text=feature).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=len(X.columns), columnspan=2)


def plot_regression():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, model.predict(X_test_scaled), alpha=0.5, color='blue', label='Actual vs Predicted')
    plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, model.predict(X_test_scaled), 1))(np.unique(y_test)), color='red', linestyle='--', label='Regression Line')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices with Regression Line")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_button = tk.Button(root, text="Plot Actual vs Predicted Prices with Regression Line", command=plot_regression)
plot_button.grid(row=len(X.columns) + 1, columnspan=2)

root.mainloop()
