# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
import pandas as pd

# Sample data
data = [
    "[2024-04-08 16:58:27.589463] Successful login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 302 0',
    '"GET /registration/ HTTP/1.1" 200 1595',
    '"GET /registration/logout/ HTTP/1.1" 302 0',
    '"GET /registration/login/ HTTP/1.1" 200 1632',
    "[2024-04-08 17:59:07.238194] Failed login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 200 1645',
    "[2024-04-08 17:59:11.629098] Failed login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 200 1645',
    "[2024-04-08 18:01:02.835081] Successful login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 302 0',
    '"GET /registration/ HTTP/1.1" 200 1595',
    '"GET /registration/logout/ HTTP/1.1" 302 0',
    '"GET /registration/login/ HTTP/1.1" 200 1632',
    '"GET /static/bootstrap.min.css HTTP/1.1" 304 0',
    '"GET /static/bootstrap.min.js HTTP/1.1" 304 0',
    '"GET /static/jquery-3.7.1.min.js HTTP/1.1" 304 0',
    "[2024-04-08 21:35:44.464759] Successful login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 302 0',
    '"GET /registration/ HTTP/1.1" 200 1595',
    '"GET /registration/logout/ HTTP/1.1" 302 0',
    '"GET /registration/login/ HTTP/1.1" 200 1632',
    "[2024-04-08 21:35:52.091581] Failed login attempt for user 'user' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 200 1645',
    '"GET /registration/signup HTTP/1.1" 200 2503',
    '"POST /registration/signup HTTP/1.1" 302 0',
    '"GET /registration/login/ HTTP/1.1" 200 1632',
    "[2024-04-08 21:36:19.955105] Successful login attempt for user 'tata' from IP 127.0.0.1",
    '"POST /registration/login/ HTTP/1.1" 302 0',
    '"GET /registration/ HTTP/1.1" 200 1595',
]

# Extracting relevant information
timestamps = []
events = []
users = []
ip_addresses = []

for line in data:
    if "Successful login" in line or "Failed login" in line:
        parts = line.split(" ")
        timestamp = parts[0][1:] + " " + parts[1][:-1]
        event = parts[2] + " " + parts[3]  # Concatenate HTTP method and endpoint
        user = parts[7][1:-2]  # Extract user without quotes
        ip_address = parts[-1]
        timestamps.append(timestamp)
        events.append(event)
        users.append(user)
        ip_addresses.append(ip_address)

# Create datasets
login_data = {
    "Timestamp": timestamps,
    "Event": events,
    "User": users,
    "IP Address": ip_addresses,
}

# Convert to DataFrame
login_df = pd.DataFrame(login_data)

# Display the datasets
print(login_df)


# Extract features and target label
X = login_df.drop(columns=["Timestamp", "Event"])
y = login_df["Event"]

# Convert categorical variables into numerical values if needed
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Initialize and train the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=200, criterion="entropy", random_state=0
)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# from sklearn.model_selection import GridSearchCV

# # Define your Random Forest classifier
# classifier = RandomForestClassifier()

# # Define the parameter grid
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "criterion": ["gini", "entropy"],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }

# # Initialize GridSearchCV with your classifier and parameter grid
# grid_search = GridSearchCV(classifier, param_grid, cv=2, n_jobs=-1)


# # Fit the grid search to your data
# grid_search.fit(X_train, y_train)

# # Get the best parameters found by the grid search
# best_params = grid_search.best_params_
