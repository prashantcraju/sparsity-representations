import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class CustomModel(nn.Module):
    def __init__(self, N, M, sparsity=0.5):
        super(CustomModel, self).__init__()
        self.N = N
        self.M = M
        self.sparsity = sparsity
        
        # Create mask for sparse connections
        self.masks = {
            'XOR': torch.rand(2*N, M) < sparsity,
            'Input1': torch.cat((torch.rand(N, M) < sparsity, torch.zeros(N, M)), dim=0),
            'Input2': torch.cat((torch.zeros(N, M), torch.rand(N, M) < sparsity), dim=0)
        }

        # Define layers
        self.fc1 = nn.Linear(2*N, M)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(M, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, task):
        x = torch.cat((x1, x2), dim=1)
        # Apply the task-specific mask
        self.fc1.weight = nn.Parameter(self.fc1.weight.data * self.masks[task].float())
        self.fc1_out = self.fc1(x)
        self.relu_out = self.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu_out)
        x = self.sigmoid(self.fc2_out)
        return x, self.fc1_out, self.relu_out, self.fc2_out

def generate_data(task, size=1000, noise_std=0.1):
    x1 = torch.randint(0, 2, (size, N)).float()
    x2 = torch.randint(0, 2, (size, N)).float()
    
    # Apply Gaussian noise
    x1 += noise_std * torch.randn_like(x1)
    x2 += noise_std * torch.randn_like(x2)
    
    if task == 'XOR':
        y = (x1[:,0] != x2[:,0]).float()
    elif task == 'Input1':
        y = x1[:,0].clone()
    elif task == 'Input2':
        y = x2[:,0].clone()

    # Ensure target values are either 0 or 1
    y = torch.clamp(y, 0, 1)
    return x1, x2, y

def compute_loss(output, target, model, beta1, beta2):
    cross_entropy_loss = nn.BCELoss()(output, target)
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param)**2
    return beta1 * cross_entropy_loss + beta2 * l2_loss

# Parameters
N = 5
M = 10
sparsity_values = [0.05 * i for i in range(1, 20)]  # Sparsity from 0.1 to 0.9
tasks = ['XOR', 'Input1', 'Input2']
# beta1, beta2 = 0.8, 0.1  # Adjust these values as needed

beta_1 = [0.05 * i for i in range(1, 20)] 
beta_2 = [0.05 * i for i in range(1, 20)]

for i in range(len(beta_1)):
    for j in range(len(beta_2)):
        beta1 = beta_1[i]
        beta2 = beta_2[j]


        
        for sparsity in sparsity_values:
            model = CustomModel(N, M, sparsity)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        
            for task in tasks:
                x1, x2, y = generate_data(task)
                losses = []
                activations = []
                plt.figure(figsize=(18, 4))  # Enlarged figure size for clarity
                
                for epoch in range(200):
                    optimizer.zero_grad()
                    outputs, fc1_out, relu_out, fc2_out = model(x1, x2, task)
                    loss = compute_loss(outputs.squeeze(), y, model, beta1, beta2)
                    loss.backward()
                    optimizer.step()
                    if epoch == 199:
                        activations = [fc1_out.detach(), relu_out.detach(), fc2_out.detach()]
                    losses.append(loss.item())
                
                # Plot losses
                plt.subplot(1, 4, 1)
                plt.plot(losses)
                plt.title(f'Loss for {task}, Sparsity {sparsity:.2f}, Beta1 {beta1:.2f} Beta2 {beta2:.2f}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
        
                # Plot final weights of fc1
                plt.subplot(1, 4, 2)
                plt.imshow(model.fc1.weight.detach().numpy(), cmap='hot', aspect='auto')
                plt.colorbar()
                plt.title(f'Weights of fc1')
                plt.xlabel('Hidden Units')
                plt.ylabel('Input Units')
        
                # Plot activations
                for i, layer in enumerate(['fc1_out', 'relu_out']):
                    plt.subplot(1, 4, i+3)
                    act = activations[i].numpy()
                    plt.imshow(act, cmap='hot', aspect='auto')
                    plt.colorbar()
                    plt.title(f'{layer} Activations')
                    plt.xlabel('Index')
                    plt.ylabel('Activation Units')
        
                plt.tight_layout()
                plt.savefig(f'{task}-Sparsity-{sparsity:.2f}-Beta1-{beta1}-Beta2-{beta2}.png')
                plt.show()
