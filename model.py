#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:14:51 2024

@author: prashantcraju
"""

import torch
import torch.nn as nn
import torch.optim as optim

class MultimodalModel(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(MultimodalModel, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(input_size2, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size * 2, 1)  # Output size is 1 for binary classification

    def forward(self, x1, x2):
        out1 = self.relu(self.fc1(x1))
        out2 = self.relu(self.fc2(x2))
        combined = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        output = self.fc3(combined)
        return torch.sigmoid(output)

def multimodal_task(input1, input2):
    # Simulate data for the task
    x1 = torch.tensor(input1, dtype=torch.float32)
    x2 = torch.tensor(input2, dtype=torch.float32)

    # Define the model and optimizer
    model = MultimodalModel(input_size1=x1.size(1), input_size2=x2.size(1), hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function with L1-norm penalty
    criterion = nn.BCELoss()
    l1_lambda = 0.01  # L1-norm penalty coefficient

    # Training loop
    for epoch in range(1000):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        output = model(x1, x2)
        loss = criterion(output, torch.ones_like(output))  # Assuming binary classification

        # Compute L1-norm regularization
        l1_regularization = torch.norm(torch.cat([param.view(-1) for param in model.parameters()]), 1)

        # Add L1-norm penalty to the loss
        loss += l1_lambda * l1_regularization

        loss.backward()
        optimizer.step()

    return output  # Return the final output tensor

# Example usage
input1 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Example input data for channel 1
input2 = [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3]]  # Example input data for channel 2

output = multimodal_task(input1, input2)
print(output)  # Print the final output tensor after training
