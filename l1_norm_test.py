import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class MultimodalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultimodalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        hidden_activity = x
        x = self.fc2(x)
        return x, hidden_activity

# Add Gaussian noise
def add_gaussian_noise(input, mean=0.0, std=0.4):
    noise = torch.randn_like(input) * std + mean
    return input + noise

# Define the custom loss function
class CustomLoss(nn.Module):
    def __init__(self, l1_lambda):
        super(CustomLoss, self).__init__()
        self.l1_lambda = l1_lambda
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target, hidden_activity):
        ce_loss = self.cross_entropy(output, target)
        l1_loss = self.l1_lambda * torch.norm(hidden_activity, p=1)
        return ce_loss + l1_loss

# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloaders['train']:
            inputs = add_gaussian_noise(inputs)
            outputs, hidden_activity = model(inputs)
            loss = criterion(outputs, labels, hidden_activity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate_model(model, dataloaders, criterion):
    model.eval()
    total_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = add_gaussian_noise(inputs)
            outputs, hidden_activity = model(inputs)
            loss = criterion(outputs, labels, hidden_activity)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    return total_loss / len(dataloaders['val']), corrects.double() / total

# Generate synthetic data
np.random.seed(0)
torch.manual_seed(0)
data_size = 1000
input_data = np.random.randint(0, 2, (data_size, 2)).astype(np.float32)

# Labels for the three tasks
labels_task1 = input_data[:, 0].astype(np.int64)
labels_task2 = input_data[:, 1].astype(np.int64)
labels_task3 = (input_data[:, 0].astype(int) ^ input_data[:, 1].astype(int)).astype(np.int64)

# Create DataLoaders
dataset_task1 = TensorDataset(torch.tensor(input_data), torch.tensor(labels_task1))
dataset_task2 = TensorDataset(torch.tensor(input_data), torch.tensor(labels_task2))
dataset_task3 = TensorDataset(torch.tensor(input_data), torch.tensor(labels_task3))

dataloaders_task1 = {
    'train': DataLoader(dataset_task1, batch_size=32, shuffle=True),
    'val': DataLoader(dataset_task1, batch_size=32, shuffle=False)
}
dataloaders_task2 = {
    'train': DataLoader(dataset_task2, batch_size=32, shuffle=True),
    'val': DataLoader(dataset_task2, batch_size=32, shuffle=False)
}
dataloaders_task3 = {
    'train': DataLoader(dataset_task3, batch_size=32, shuffle=True),
    'val': DataLoader(dataset_task3, batch_size=32, shuffle=False)
}

# Experiment with different L1 norm values for each task
l1_values = [i*2 for i in range(50)]
performance_task1 = []
performance_task2 = []
performance_task3 = []

# Function to run experiment for a given task
def run_experiment(dataloaders, performance_list):
    for l1_lambda in l1_values:
        model = MultimodalModel(input_size=2, hidden_size=10, output_size=2)
        criterion = CustomLoss(l1_lambda=l1_lambda)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
        loss, acc = evaluate_model(model, dataloaders, criterion)
        performance_list.append(acc.item())

run_experiment(dataloaders_task1, performance_task1)
run_experiment(dataloaders_task2, performance_task2)
run_experiment(dataloaders_task3, performance_task3)

# Plot the results for all tasks on the same plot
plt.figure(figsize=(10, 6))

plt.scatter(l1_values, performance_task1, label='Task 1: Input 1 Classification', color='r')
plt.scatter(l1_values, performance_task2, label='Task 2: Input 2 Classification', color='g')
plt.scatter(l1_values, performance_task3, label='Task 3: XOR Classification', color='b')

plt.xlabel('L1 Norm Value')
plt.ylabel('Performance (Accuracy)')
plt.title('Performance vs. L1 Norm Value for Different Tasks')
plt.legend()
plt.grid(True)
plt.show()
