import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
data = pd.read_csv('fileless_malware_dataset.csv')

# Preprocessing
X = data.drop('is_malware', axis=1)
y = data['is_malware']

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Simulate new data (for demonstration purposes, using random values)
# In a real-world scenario, you'd gather this data from actual software behavior
new_data = np.random.rand(1, X_train.shape[1])

# Normalize the new data using the same scaler used for the training data
new_data_normalized = scaler.transform(new_data)

# Reshape the data to match the input shape expected by the model
new_data_reshaped = np.reshape(new_data_normalized, (1, new_data_normalized.shape[1], 1))

# Predict using the trained model
prediction = model.predict(new_data_reshaped)
predicted_class = "Malware" if prediction >= 0.5 else "Benign"

# Display the predicted class
print(f"The predicted class for the new data is: {predicted_class}")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
