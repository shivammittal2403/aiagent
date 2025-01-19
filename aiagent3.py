import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import random

# Sample function to generate CFG data from a malware binary
def generate_CFG(binary_path, num_samples=1000):
    data_list = []
    for _ in range(num_samples):
        num_nodes = random.randint(5, 20)
        nodes = torch.randn((num_nodes, 2))
        edges = torch.randint(0, num_nodes, (2, random.randint(num_nodes-1, num_nodes * (num_nodes-1) // 2)))
        
        # Randomly assign a Label (0: benign, 1: metamorphic malware)
        label = torch.tensor([random.randint(0, 1)], dtype=torch.long)
        
        data = Data(x=nodes, edge_index=edges, y=label)
        data_list.append(data)
    
    return data_list

# Define the GNN model
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

# Generate CFG data from hypothetical malware binaries
dataset = generate_CFG("path_to_malware_binary")

# Load data
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train the GNN model
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(100):
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        
        # Adjust the Loss calculation for batched graph data using the batch attribute
        y_true = data.y[data.batch]
        loss = F.nll_loss(out, y_true)
        
        loss.backward()
        optimizer.step()

# Inference on a single sample
model.eval()
sample_data = dataset[0]  # Assuming you're taking the first sample from the dataset
out = model(sample_data)

# Get the predicted class
pred = out.max(dim=1)[1]

# Check if the first node in the binary is detected as metamorphic malware
if pred[0].item() == 1:
    print("The first node in the binary is detected as metamorphic malware!")
else:
    print("The first node in the binary is benign.")
