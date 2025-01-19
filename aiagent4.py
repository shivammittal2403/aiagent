from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Features: [Heap Size, Stack Size, Threads, API Calls, Loaded DLLs, Timers, Semaphores, Files, Sockets, Events, IPC]

# Simulated memory dump data
X = np.array([
    [200, 50, 2, 5, 1, 0, 1, 2, 8, 8, 0],  # benign
    [220, 55, 2, 4, 1, 0, 1, 2, 0, 0, 0],  # benign
    [500, 200, 10, 20, 5, 2, 3, 5, 2, 1, 1],  # malicious
    [210, 52, 2, 5, 1, 0, 1, 2, 0, 0, 0],  # benign
    [488, 190, 9, 18, 4, 2, 3, 4, 2, 1, 1],  # malicious
])

# Labels: 0 for benign, 1 for malicious
y = np.array([0, 0, 1, 0, 1])

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
