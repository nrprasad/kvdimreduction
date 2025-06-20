import torch
import torch.nn as nn

# Select device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PowerLinearModel(nn.Module):
    def __init__(self, k, L):
        super().__init__()
        self.k = k
        self.L = L
        dim = 2 ** k
        layers = []
        for _ in range(L):
            layers.append(nn.Linear(dim, dim))
        self.layers = nn.ModuleList(layers)
        # Initialize all layers' weights and biases to zero
        # for layer in self.layers:
        #    nn.init.zeros_(layer.weight)
        #    if layer.bias is not None:
        #        nn.init.zeros_(layer.bias)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss_fn(self, x, target, reg_lambda=5e-3):
        """
        Computes the loss as the sum of squared error between model output and target,
        plus L1 regularization on all weights in all linear layers.

            Args:
                x: Input tensor.
                target: Target tensor.
                reg_lambda: Regularization strength (float).

            Returns:
                loss: Scalar tensor.
        """
        output = self.forward(x)
        squared_error = torch.sum((output - target) ** 2)
        l1_reg = 0.0
        for layer in self.layers:
            l1_reg = l1_reg + torch.sum(torch.abs(layer.weight))
        loss = squared_error + reg_lambda * l1_reg
        return loss

def hadamard_matrix(n, device=None, dtype=None):
    # Iterative construction of Hadamard matrix
    if device is None:
        device = torch.device("cpu")
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    for _ in range(int(torch.log2(torch.tensor(n)).item())):
        H = torch.cat([torch.cat([H, H], dim=1),
                     torch.cat([H, -H], dim=1)], dim=0)
    return H

def hadamard_transform_batch(batch_size, k, Hadamard_matrix ,device=None, dtype=None):
    """
    Returns two tensors X and Y of shape (B, 2^k).
    X: random tensor, each row is a random vector of dimension 2^k.
    Y: each row is the Hadamard transform of the corresponding row of X.
    """
    n = 2 ** k
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    X = torch.randn(batch_size, n, device=device, dtype=dtype)
    Y = X @ Hadamard_matrix
    return X, Y

def train_model(model:PowerLinearModel, train_iterations, batch_size, lr=1e-3):
    k = model.k
    L = model.L
    model.train()

    n = 2 ** k
    dtype = torch.float32
    hadamard_mat = hadamard_matrix(n, device=device, dtype=dtype)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for step in range(train_iterations):
        # Generate batch data on the correct device
        X, Y = hadamard_transform_batch(batch_size, k, hadamard_mat, device=device)
        # Forward and loss
        loss = model.loss_fn(X, Y)
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 1000 == 0:
            print(f"Step {step+1}/{train_iterations}, Loss: {loss.item():.4f}")

def print_model(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Layer: {name}")
            print("Weight:")
            print(module.weight.data)
            print("-" * 40)

def validate_model(model:PowerLinearModel, batch_size, device=device):
    k = model.k
    L = model.L
    n = 2 ** k
    dtype = torch.float32
    hadamard_mat = hadamard_matrix(n, device=device, dtype=dtype)

    # Computing the errors 
    X, Y = hadamard_transform_batch(batch_size, k, hadamard_mat, device=device)
    output = model.forward(X)
    avg_squared_error = torch.sum((output - Y) ** 2)/(batch_size*(2 ** k))
    print(f"Average squared error: {avg_squared_error}")

    l1_reg = 0.0
    for layer in model.layers:
        l1_reg = l1_reg + torch.sum(torch.abs(layer.weight))
    avg_l1_reg = l1_reg/((2**k)*L)

    print(f"Total L1 error: {l1_reg}")

 
# Hyperparameters
k = 3  # so 2^k = 8
L = 3
train_iterations = 1000000  # number of training iterations
batch_size = 128
lr = 1e-4

def main():
    # Create model instance, move to device and train it
    model = PowerLinearModel(k,L)
    model.to(device)
    train_model(model, train_iterations, batch_size,lr)



    validate_model(model,batch_size,device)
    # Print model parameters
    print_model(model)

if __name__ == "__main__":
    main()

