import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class MultimodalModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultimodalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.relu(self.fc1(x1))
        x2 = self.relu(self.fc1(x2))
        out1 = self.fc2(x1)
        out2 = self.fc2(x2)
        return out1, out2
        
# Define functions for training model, computing RDM, and plotting RDM
def train_model(samples, task, sparsity_level):
    input_size = samples.shape[1]
    output_size = 2  # Output size of the model
    model = MultimodalModel(input_size, output_size)
    
    # Loss function with L1 regularization
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    representations = []
    for _ in range(epochs):
        # Generate random indices for sparsity
        indices = np.random.rand(*samples.shape) > sparsity_level
        input1 = torch.tensor(samples * indices, dtype=torch.float32)
        input2 = torch.tensor(samples * (~indices), dtype=torch.float32)

        optimizer.zero_grad()
        out1, out2 = model(input1, input2)

        # Compute L1 loss
        loss = criterion(out1, out2) + sparsity_level * torch.norm(model.fc1.weight, 1)
        loss.backward()
        optimizer.step()
        
        representations.append(out1.detach().numpy())  # Detach and convert to numpy array
    
    # Reshape representations to 2D array
    representations = np.array(representations).reshape(-1, output_size)
    return representations

def compute_rdm(representations):
    distances = pdist(representations, metric='euclidean')
    rdm = squareform(distances)
    return rdm

def multimodal_task(samples, sparsity_level):
    representations = train_model(samples, 'xor', sparsity_level)
    rdm = compute_rdm(representations)
    return rdm

# Generate synthetic data for two distinct inputs
samples_input1 = np.random.rand(100, 2)  # 100 samples for input 1
samples_input2 = np.random.rand(100, 2)  # 100 samples for input 2

# Define parameters
epochs = 200
sparsity_levels = [0.2, 0.4, 0.6, 0.8]

# Initialize subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Run tasks for each sparsity level and plot RDMs
for i, sparsity_level in enumerate(sparsity_levels):
    print(f"Running for sparsity level: {sparsity_level}")
    rdm_input1 = multimodal_task(samples_input1, sparsity_level)
    rdm_input2 = multimodal_task(samples_input2, sparsity_level)
    
    # Combine RDMs for both inputs using XOR (logical XOR)
    rdm_xor = np.logical_xor(rdm_input1, rdm_input2)
    
    # Determine subplot position
    row = i // 2
    col = i % 2
    
    # Plot RDM in corresponding subplot
    axs[row, col].imshow(rdm_xor, cmap='viridis')
    axs[row, col].set_title(f'Sparsity Level: {sparsity_level}')
    axs[row, col].set_xlabel('Samples')
    axs[row, col].set_ylabel('Samples')
    axs[row, col].tick_params(axis='both', which='both', bottom=False, left=False)

# Adjust spacing between subplots
plt.tight_layout()

# Show plot
plt.show()
