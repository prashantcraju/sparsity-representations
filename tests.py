#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:17:01 2024

@author: prashantcraju
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MultimodalModel(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(MultimodalModel, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(input_size2, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size * 2, 2)  # Output size is 2 for visualization purposes

    def forward(self, x1, x2):
        out1 = self.relu(self.fc1(x1))
        out2 = self.relu(self.fc2(x2))
        combined = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        output = self.fc3(combined)
        return output

def generate_sparse_data(num_samples, input_dim, sparsity_level):
    data = torch.randn(num_samples, input_dim)
    mask = torch.rand(num_samples, input_dim) < sparsity_level
    data[mask] = 0  # Apply sparsity mask
    return data

def plot_representations(x1, x2, representations, sparsity_level):
    plt.figure(figsize=(10, 5))
    plt.scatter(representations[:, 0], representations[:, 1], c='b', label='Representation')
    plt.scatter(x1[:, 0], x1[:, 1], c='r', label='Input1')
    plt.scatter(x2[:, 0], x2[:, 1], c='g', label='Input2')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title(f'Multimodal Representational Geometry (Sparsity Level: {sparsity_level})')
    plt.show()

def multimodal_task(input1_samples, input2_samples, sparsity_level, epochs=1000):
    # Generate sparse data
    x1 = generate_sparse_data(input1_samples, 2, sparsity_level)
    x2 = generate_sparse_data(input2_samples, 2, sparsity_level)

    # Define the model and optimizer
    model = MultimodalModel(input_size1=x1.size(1), input_size2=x2.size(1), hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    representations = []
    for epoch in range(epochs):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        output = model(x1, x2)
        
        # Store representations for visualization
        representations.append(output.detach().numpy())

        loss = torch.norm(output, 2)  # Just an example loss for visualization

        loss.backward()
        optimizer.step()

    representations = torch.tensor(representations).squeeze()  # Convert to tensor and remove singleton dimension
    plot_representations(x1, x2, representations, sparsity_level)

# Example usage with different parameters
input1_samples = 100
input2_samples = 100
sparsity_levels = [(i*.05) for i in range(20)]  # Sparsity levels to test
epochs = 20000  # Increase training epochs for better convergence and representation learning

for sparsity in sparsity_levels:
    multimodal_task(input1_samples, input2_samples, sparsity, epochs)
