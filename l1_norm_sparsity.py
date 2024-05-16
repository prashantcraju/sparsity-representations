import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CustomModel(nn.Module):
    def __init__(self, N, M):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(2 * N, M)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(M, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        output = self.sigmoid(self.fc2(relu_out))
        return output, relu_out

def compute_loss(outputs, targets, model, beta1, beta2):
    criterion = nn.BCELoss()
    cross_entropy_loss = criterion(outputs, targets)
    l1_norm = sum(torch.sum(torch.abs(param)) for param in model.fc1.parameters())  # Apply L1 only on fc1 weights
    return (beta1 * cross_entropy_loss) + (beta2 * l1_norm)

def generate_input1_discrimination_data(N, batch_size=100):
    x1 = torch.randint(0, 2, (batch_size, N)).float()  # Varying input 1
    x2 = torch.randint(0, 2, (batch_size, N)).float()  # Irrelevant input 2
    y = x1[:, 0]  # Output depends only on the first bit of Input 1
    return torch.cat((x1, x2), dim=1), y

def generate_input2_discrimination_data(N, batch_size=100):
    x1 = torch.randint(0, 2, (batch_size, N)).float()  # Irrelevant input 1
    x2 = torch.randint(0, 2, (batch_size, N)).float()  # Varying input 2
    y = x2[:, 0]  # Output depends only on the first bit of Input 2
    return torch.cat((x1, x2), dim=1), y

def generate_xor_data(N, batch_size=100):
    x = torch.randint(0, 2, (batch_size, 2 * N)).float()  # Generate XOR pairs
    y = (x[:, 0] != x[:, 1]).float().unsqueeze(1)  # XOR operation
    return x, y

N = 1
M = 10
model = CustomModel(N, M)
optimizer = optim.Adam(model.parameters(), lr=0.001)
sparsity_levels = [i * 0.05 for i in range(20)]

for sparsity in sparsity_levels:
    x, y = generate_input2_discrimination_data(N)
    optimizer.zero_grad()
    outputs, relu_out = model(x)
    loss = compute_loss(outputs.squeeze(), y, model, beta1=0.85, beta2=sparsity)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        _, relu_out = model(x)
        scaler = StandardScaler()
        activations = scaler.fit_transform(relu_out.numpy())
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(activations)

        plt.figure(figsize=(4, 4))
        for i, (pc1, pc2) in enumerate(zip(pca_results[:, 0], pca_results[:, 1])):
            label = f'Input 1: {int(x[i, 0])}, Input 2: {int(x[i, 1])}'
            plt.scatter(pc1, pc2, alpha=0.7)
            plt.text(pc1, pc2, label, fontsize=9)  # Adjust fontsize as needed

        plt.title(f'PCA of ReLU Activations for Input 2 Discrimination (Sparsity {sparsity * 100:.000f}%)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.savefig(f'__input2_sparsity_{sparsity:.2f}_constrained_beta_1.85_beta_2_.2.jpg',fontsize = 6)
        plt.show()