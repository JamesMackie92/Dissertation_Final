import tkinter as tk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# load CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\halfa\Documents\seattle_weather.csv")

# Define the features (X) and target variable (y)
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']

# Initialize the Random Forest classifier and train it with the entire dataset
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Define function for making predictions
def predict_weather():
    # Retrieve input values from entry widgets
    precipitation = float(entry_precipitation.get())
    temp_max = float(entry_temp_max.get())
    temp_min = float(entry_temp_min.get())
    wind = float(entry_wind.get())

    # Make prediction using the trained model
    prediction = rf_classifier.predict([[precipitation, temp_max, temp_min, wind]])

    # Display the prediction
    label_prediction.config(text=f"Predicted Weather: {prediction[0]}")

# Create GUI window
root = tk.Tk()
root.title("Weather Prediction")

# Create entry widgets for input features
entry_precipitation = tk.Entry(root)
entry_precipitation.grid(row=0, column=1)
entry_temp_max = tk.Entry(root)
entry_temp_max.grid(row=1, column=1)
entry_temp_min = tk.Entry(root)
entry_temp_min.grid(row=2, column=1)
entry_wind = tk.Entry(root)
entry_wind.grid(row=3, column=1)

# Create labels for input features
tk.Label(root, text="Precipitation (inches):").grid(row=0, column=0)
tk.Label(root, text="Max Temperature (F):").grid(row=1, column=0)
tk.Label(root, text="Min Temperature (F):").grid(row=2, column=0)
tk.Label(root, text="Wind Speed (mph):").grid(row=3, column=0)

# Create button for making predictions
button_predict = tk.Button(root, text="Predict Weather", command=predict_weather)
button_predict.grid(row=4, column=0, columnspan=2)

# Create label for displaying prediction
label_prediction = tk.Label(root, text="")
label_prediction.grid(row=5, column=0, columnspan=2)

# Run the GUI application
root.mainloop()
