from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Simulated executable file data (for demonstration purposes)
# Features: [File Size, Number of Sections, Entropy, Suspicious APIS]
data = {
    'File Size': [2000, 2200, 2500, 2750, 8000, 8588, 9000, 9500],
    'Num Sections': [3, 3, 4, 4, 8, 8, 9, 9],
    'Entropy': [6.5, 6.6, 6.7, 6.8, 7.5, 7.6, 7.7, 7.8],
    'Suspicious_APIs': [0, 1, 0, 1, 5, 6, 5 , 6],
    'Is Malware': [0, 0, 0, 0, 1, 1, 1, 1]  # 0 for benign, 1 for malware
}

df = pd.DataFrame(data)

# Features and Labels
X = df[['File Size', 'Num Sections', 'Entropy', 'Suspicious_APIs']]
y = df['Is Malware']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=50)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Now you can use clf.predict() to make predictions on new executable file data
