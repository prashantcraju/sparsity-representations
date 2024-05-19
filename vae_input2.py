import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, output_size, batch_first=True)
    
    def forward(self, z, seq_len):
        hidden = self.fc(z).unsqueeze(0)
        output, _ = self.rnn(hidden.repeat(seq_len, 1, 1).permute(1, 0, 2))
        return output

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z, x.size(1))
        return x_recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, lambda_l1):
        BCE = nn.BCEWithLogitsLoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        L1 = lambda_l1 * torch.sum(torch.abs(mu))
        return BCE + KLD + L1

def generate_input2_discrimination_data_with_noise(size, noise_level=0.1):
    x1 = torch.randint(0, 2, (size, 1, 1), dtype=torch.float32)
    x2 = torch.randint(0, 2, (size, 1, 1), dtype=torch.float32)
    y = x2.view(-1)  # Target depends only on x1

    # Add Gaussian noise to inputs
    noise1 = torch.randn(x1.shape) * noise_level
    noise2 = torch.randn(x2.shape) * noise_level
    x1_noisy = x1 + noise1
    x2_noisy = x2 + noise2

    return x1_noisy, x2_noisy, y

def train_vae(model, x1_noisy, x2_noisy, lambda_l1, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    x = torch.cat((x1_noisy, x2_noisy), dim=-1)  # Concatenate along the feature dimension

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = model.loss_function(x_recon, x, mu, logvar, lambda_l1)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    _, hidden = model.encoder(x)
    return hidden.detach()

# Example model and data setup
x1_noisy, x2_noisy, y = generate_input2_discrimination_data_with_noise(100)

# Ensure y is properly shaped and is a PyTorch tensor
if not isinstance(y, torch.Tensor):
    y = torch.tensor(y, dtype=torch.float32)
y = y.view(-1)  # Ensure y is flattened to [batch_size]

# Threshold noisy inputs to determine binary states
x1_binary = (x1_noisy > 0.5).float()
x2_binary = (x2_noisy > 0.5).float()

# Define the VAE model parameters
input_size = 2
hidden_size = 10
latent_size = 2

# Create the VAE model
vae = VAE(input_size, hidden_size, latent_size)

# L1 norm value
lambda_l1 = 0.05

# Train the VAE model
hidden_states = train_vae(vae, x1_noisy, x2_noisy, lambda_l1)

# Perform PCA on the hidden states
pca = PCA(n_components=2)
reduced = pca.fit_transform(hidden_states.numpy())

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

title = f'Input 2 PCA of VAE Latent Space with L1 Norm = {lambda_l1}'
plt.title(title)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(title+'.png', bbox_inches='tight')
plt.legend()
plt.show()
