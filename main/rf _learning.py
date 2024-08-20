import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame, dataset will be in file along with code, change path accordingly.
df = pd.read_csv(r"C:\Users\halfa\Documents\seattle_weather.csv")

# Define the features (X) and target variable (Y)
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']

# Initialize lists to store accuracies and random states
accuracies = []
random_states = range(10)  # Random states from 0 to 9

# Iterate over each random state
for random_state in random_states:
    # Split the data into training and testing sets using the current random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Train the model on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing
    y_pred = rf_classifier.predict(X_test)

    # Calculate the accuracy of the model and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(random_states, accuracies, marker='o')
plt.title('Accuracy of Random Forest Classifier (10 Random States)')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.xticks(range(10))
plt.grid(True)
plt.show()
