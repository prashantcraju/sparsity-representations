import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

class MultimodalRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultimodalRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        output, hidden = self.rnn(x)
        output = self.relu(output)
        output = self.sigmoid(self.fc(output))
        return output.view(-1), hidden.squeeze(0)  # Ensure output is flattened to [batch_size]

    def l1_regularization(self, hidden):
        return torch.sum(torch.abs(hidden))

def generate_input1_discrimination_data_with_noise(size, noise_level=0.1):
    x1 = torch.randint(0, 2, (size, 1, 1), dtype=torch.float32)
    x2 = torch.randint(0, 2, (size, 1, 1), dtype=torch.float32)
    y = x1.view(-1)  # Target depends only on x1

    # Add Gaussian noise to inputs
    noise1 = torch.randn(x1.shape) * noise_level
    noise2 = torch.randn(x2.shape) * noise_level
    x1_noisy = x1 + noise1
    x2_noisy = x2 + noise2

    return x1_noisy, x2_noisy, y

def train_model(lambda_l1, model, x1_noisy, x2_noisy, y, num_epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    # Debugging: Print y shape and content before training
    print(f'Initial y shape: {y.shape}, y content: {y}, y type: {type(y)}')

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs, hidden = model(x1_noisy, x2_noisy)

        # Ensure outputs are also correctly reshaped
        outputs = outputs.view(-1)  # Ensuring this is flattened to [batch_size]

        # Debugging: Print shapes to ensure they match
        if epoch == 0:
            print(f'Epoch {epoch}: Outputs shape: {outputs.shape}, y shape: {y.shape}')

        loss = criterion(outputs, y)
        l1_loss = lambda_l1 * model.l1_regularization(hidden)
        total_loss = loss + l1_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item()}, L1 Loss: {l1_loss.item()}')

    return hidden.detach()

# Example model and data setup
x1_noisy, x2_noisy, y = generate_input1_discrimination_data_with_noise(100)

# Ensure y is properly shaped and is a PyTorch tensor
if not isinstance(y, torch.Tensor):
    y = torch.tensor(y, dtype=torch.float32)
y = y.view(-1)  # Ensure y is flattened to [batch_size]

# Threshold noisy inputs to determine binary states
x1_binary = (x1_noisy > 0.5).float()
x2_binary = (x2_noisy > 0.5).float()

# Fixed L1 norm value
lambda_l1 = 0.95

print(f"Training with L1 norm value: {lambda_l1}")
model = MultimodalRNN(input_size=2, hidden_size=10)
final_hidden_states = train_model(lambda_l1, model, x1_noisy, x2_noisy, y)

# Perform PCA on the hidden states
pca = PCA(n_components=2)
reduced = pca.fit_transform(final_hidden_states.numpy())

# Calculate the average of the principal components for each condition
conditions = [(1, 0), (0, 1), (1, 1), (0, 0)]
condition_means = []
for condition in conditions:
    indices = np.where((x1_binary.numpy()[:, -1, 0] == condition[0]) & (x2_binary.numpy()[:, -1, 0] == condition[1]))[0]
    if len(indices) > 0:
        mean_pc = np.mean(reduced[indices], axis=0)
        condition_means.append(mean_pc)
    else:
        condition_means.append(np.array([np.nan, np.nan]))  # Placeholder for empty conditions

# Convert condition_means to numpy array for distance calculation
condition_means = np.array(condition_means)

# Filter out invalid condition means
valid_means = condition_means[~np.isnan(condition_means).any(axis=1)]

# Calculate pairwise distances and find the nearest neighbors
distances = distance.cdist(valid_means, valid_means, 'euclidean')
nearest_neighbors = np.argsort(distances, axis=1)[:, 1:4]

# Plotting the averages with nearest neighbor lines
plt.figure(figsize=(6, 4))
for i, mean in enumerate(valid_means):
    plt.scatter(mean[0], mean[1], s=100, label=f'Condition {conditions[i]}')
    for neighbor in nearest_neighbors[i]:
        plt.plot([mean[0], valid_means[neighbor][0]], [mean[1], valid_means[neighbor][1]], 'grey', alpha=0.5)
title = f'Input 1: PCA with L1 Norm = {lambda_l1}'
plt.title(f'Input 1: PCA with L1 Norm = {lambda_l1}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(title+'.png', bbox_inches='tight')
plt.legend()
plt.show()