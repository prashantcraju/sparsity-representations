import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MultimodalModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(MultimodalModel, self).__init__()
        self.input_layer1 = nn.Linear(input_dim1, hidden_dim)
        self.input_layer2 = nn.Linear(input_dim2, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activations = {}

    def forward(self, x1, x2):
        x1 = self.input_layer1(x1)
        x2 = self.input_layer2(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = F.relu(self.hidden_layer(x))
        output = torch.sigmoid(self.output_layer(x))
        return output

    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        self.input_layer1.register_forward_hook(get_activation('input_layer1'))
        self.input_layer2.register_forward_hook(get_activation('input_layer2'))
        self.hidden_layer.register_forward_hook(get_activation('hidden_layer'))
        self.output_layer.register_forward_hook(get_activation('output_layer'))

# Custom Binary Cross Entropy Loss with L1 Regularization
class CustomLoss(nn.Module):
    def __init__(self, weight=0.01):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, predictions, targets, model):
        bce_loss = F.binary_cross_entropy(predictions, targets)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        total_loss = bce_loss + self.weight * l1_loss
        return total_loss

def plot_layer_outputs(model, x1, x2, title):
    num_layers = len(model.activations) + 2
    fig, axs = plt.subplots(num_layers, 1, figsize=(6, 2 * num_layers), squeeze=False)
    fig.suptitle(title)

    # Plot input x1
    img = axs[0, 0].imshow(x1.numpy(), cmap='gray', aspect='auto')
    axs[0, 0].set_title('Input x1')
    axs[0, 0].axis('off')
    fig.colorbar(img, ax=axs[0, 0], orientation='vertical')
    
    # Plot input x2
    img = axs[1, 0].imshow(x2.numpy(), cmap='gray', aspect='auto')
    axs[1, 0].set_title('Input x2')
    axs[1, 0].axis('off')
    fig.colorbar(img, ax=axs[1, 0], orientation='vertical')
    
    # Forward pass to get activations
    model(x1, x2)
    
    # Plot activations from each layer
    i = 2
    for name, activation in model.activations.items():
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
        elif activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        
        img = axs[i, 0].imshow(activation.cpu().numpy(), cmap='gray', aspect='auto')
        axs[i, 0].set_title(name)
        axs[i, 0].axis('off')
        fig.colorbar(img, ax=axs[i, 0], orientation='vertical')
        i += 1
    
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()

    
def generate_xor_data(batch_size, sparsity=0.5):
    """
    Generate XOR data with a given level of sparsity.

    Parameters:
    - batch_size (int): Number of samples to generate.
    - sparsity (float): Fraction of inputs to be set to zero (0 to 1).

    Returns:
    - Tuple of torch.Tensor: Inputs x1, x2 and targets y.
    """
    # Generate deterministic XOR pairs
    x1 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).repeat(batch_size // 4, 1)
    x2 = torch.tensor([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=torch.float32).repeat(batch_size // 4, 1)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).repeat(batch_size // 4, 1)

    # Apply sparsity by setting a randomly selected subset of inputs to zero
    mask_x1 = torch.rand(x1.shape) < sparsity  # Generate a mask where entries less than sparsity are True
    mask_x2 = torch.rand(x2.shape) < sparsity

    x1[mask_x1] = 0  # Apply the mask to x1
    x2[mask_x2] = 0  # Apply the mask to x2

    return x1, x2, y


for i in range(6):
    # Instantiate model and loss
    model = MultimodalModel(input_dim1=2, input_dim2=2, hidden_dim=10)
    model.register_hooks()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Example usage
    batch_size = 20  # Generate 20 samples
    sparsity = .2*i   # 30% of the data will be zeros
    x1, x2, targets = generate_xor_data(batch_size, sparsity)
        
        
        # Train and visualize periodically
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = CustomLoss()(outputs, targets, model)
        loss.backward()
        optimizer.step()
        if epoch % 250 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            plot_layer_outputs(model, x1, x2,  f'XOR Task Sparsity {sparsity}')  # Visualize layer outputs
