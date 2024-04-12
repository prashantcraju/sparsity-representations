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

def multimodal_task(input1_samples, input2_samples, input1_dim, input2_dim, sparsity_levels, epochs=1000):
    # Initialize empty list to store representations for each sparsity level
    all_representations = []

    # Define the model and optimizer
    model = MultimodalModel(input_size1=input1_dim, input_size2=input2_dim, hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for sparsity in sparsity_levels:
        # Generate sparse data for each input channel
        x1 = generate_sparse_data(input1_samples, input1_dim, sparsity)
        x2 = generate_sparse_data(input2_samples, input2_dim, sparsity)

        # Training loop for each sparsity level
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x1, x2)
            loss = torch.norm(output, 2)  # Just an example loss for visualization
            loss.backward()
            optimizer.step()

        # Store representations for this sparsity level
        all_representations.append(output.detach().numpy())

    # Convert representations to a tensor and remove singleton dimension
    all_representations = torch.tensor(all_representations).squeeze()
    return all_representations

def plot_all_representations(all_representations, sparsity_levels):
    plt.figure(figsize=(10, 5))
    for i in range(len(sparsity_levels)):
        plt.scatter(all_representations[i][:, 0], all_representations[i][:, 1], label=f'Sparsity {sparsity_levels[i]}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title('Multimodal Representational Geometry vs. Sparsity Level')
    plt.show()

# Example usage with different parameters
input1_samples = 100
input2_samples = 100
input1_dim = 2  # Input dimension for channel 1
input2_dim = 3  # Input dimension for channel 2
sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Sparsity levels to test
epochs = 2000  # Increase training epochs for better convergence and representation learning

representations = multimodal_task(input1_samples, input2_samples, input1_dim, input2_dim, sparsity_levels, epochs)
plot_all_representations(representations, sparsity_levels)
