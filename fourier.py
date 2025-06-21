import torch
import torch.nn as nn

# Select device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PowerLinearModel(nn.Module):
    def __init__(self, dim, L, dropout_p = None):
        super().__init__()
        self.dim = dim
        self.L = L
        self.dropout_p = dropout_p
        layers = []
        for _ in range(L):
            layers.append(nn.Linear(self.dim, self.dim))
        self.layers = nn.ModuleList(layers)
        # Initialize all layers' weights and biases to zero
        # for layer in self.layers:
        #    nn.init.zeros_(layer.weight)
        #    if layer.bias is not None:
        #        nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.dropout_p is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            for layer in self.layers:
                W = layer.weight
                if self.training:
                    # BUG: should be self.dropout_p, not self.drop_p
                    mask = torch.bernoulli((1 - self.dropout_p) * torch.ones_like(W))
                    W = W * mask
                    x = x @ W.T
                else:
                    # In eval mode, should use the expected value (scale weights)
                    x = x @ ((1 - self.dropout_p) * W.T)
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
        """
            It appears that reg_lambda = (5e-3)*(2^k/8) seems to work well,
            not sure, I have only checked a few runs.
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

def linear_transform_batch(batch_size, transform_matrix ,device=None, dtype=None):
    """
    Returns two tensors X and Y of shape (B, 2^k).
    X: random tensor, each row is a random vector of dimension 2^k.
    Y: each row is the Hadamard transform of the corresponding row of X.
    """
    n = transform_matrix.shape[1]
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    X = torch.randn(batch_size, n, device=device, dtype=dtype)
    # BUG: T is undefined, should use transform_matrix
    Y = X @ transform_matrix
    return X, Y


def train_model(model, train_iterations, transform_matrix ,batch_size, lr=1e-3):
    dim = model.dim
    L = model.L
    model.train()

    dtype = torch.float32

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for step in range(train_iterations):
        # Generate batch data on the correct device
        X, Y = linear_transform_batch(batch_size,transform_matrix,device=device,dtype=dtype)
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
            print(module.weight.grad)
            print("-" * 40)


def validate_model(model, transform_matrix, batch_size, device=device):
    dim = model.dim
    L = model.L
    dtype = torch.float32
  
    # Computing the errors 
    X, Y = linear_transform_batch(batch_size, transform_matrix, device=device, dtype=dtype)
    output = model.forward(X)
    avg_squared_error = torch.sum((output - Y) ** 2)/(batch_size * transform_matrix.dim)
    print(f"Average squared error: {avg_squared_error}")

    l1_reg = 0.0
    for layer in model.layers:
        l1_reg = l1_reg + torch.sum(torch.abs(layer.weight))
    # avg_l1_reg = l1_reg/((2**k)*L)
    avg_l1_reg = l1_reg/(transform_matrix.dim * L)

    print(f"Total L1 error: {l1_reg}")

 
# Hyperparameters
k = 3  # so 2^k = 8
L = 3
train_iterations = 1000000  # number of training iterations
batch_size = 256
lr = 1e-4

def main():
    # # Create model instance, move to device and train it

    dim = 2**k
    transform_matrix = hadamard_matrix(dim,device,dtype=torch.float32)
    model = PowerLinearModel(dim,L)
    model.to(device)
    train_model(model, train_iterations, transform_matrix, batch_size, lr)
    validate_model(model, transform_matrix, batch_size,device)
    # # Print model parameters
    print_model(model)

if __name__ == "__main__":
    main()
