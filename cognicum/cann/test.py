import tinygrad.nn as nn
from tinygrad import Tensor
import numpy as np

# The classes you defined earlier go here (MultiHeadAttention2D, ResidualAttentionBlock, CANNEncoder, etc.)
from cann import CANNAutoencoder

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params  # List of model parameters to update
        self.lr = lr  # Learning rate
        self.beta1, self.beta2 = betas  # Betas for momentum terms
        self.eps = eps  # Small value to avoid division by zero
        self.t = 0  # Time step
        # Initialize first and second moment vectors
        self.m = [Tensor.zeros_like(p) for p in params]  # First moment vector (momentum)
        self.v = [Tensor.zeros_like(p) for p in params]  # Second moment vector (squared gradient)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue  # If no gradient, skip the parameter

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update the parameters
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        # Reset gradients to zero
        for p in self.params:
            p.grad = None

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

class DataLoader:
    def __init__(self, data, batch_size):
        self.data = data  # The data is expected to be a list or numpy array-like object or tensor
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            print("slicing")
            batch = self.data[i:i + self.batch_size]
            print(np.array(batch).shape)
            # Convert the batch into a tinygrad Tensor
            yield batch

def train(autoencoder, data_loader, epochs=10, lr=1e-3):
    # Initialize Adam optimizer
    params = []  # This should contain all model parameters (weights of convs, etc.)
    for param in params:
        param.requires_grad = True
    optimizer = Adam(params, lr=lr)

    for epoch in range(epochs):
        print("Epoch:", epoch, "===============================================")
        epoch_loss = 0
        batch_cnt = 0
        for batch in data_loader:
            print("Batch:", batch_cnt, "===============================================")
            input_data = batch  # This is your input data (e.g., a batch of images)

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass through the autoencoder
            output_data = autoencoder.forward(input_data)

            # Compute the reconstruction loss (MSE)
            loss = mse_loss(output_data, input_data)

            # Backward pass (compute gradients)
            loss.backward()

            # Update parameters using the optimizer
            optimizer.step()

            epoch_loss += loss.item()

        # Print loss after each epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader)}')

# Example data
data = Tensor.randn(1000, 3, 64, 64)
data_loader = DataLoader(data, batch_size=32)

# Example autoencoder configuration
encoder_config = [
    {'in_channels': 3, 'out_channels': 64, 'n_heads': 3, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'in_channels': 64, 'out_channels': 128, 'n_heads': 4, 'kernel_size': 3, 'stride': 2, 'padding': 1},
]

decoder_config = [
    {'in_channels': 128, 'out_channels': 64, 'n_heads': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'in_channels': 64, 'out_channels': 3, 'n_heads': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1},
]

# Initialize the autoencoder
autoencoder = CANNAutoencoder(encoder_config, decoder_config)

# Train the autoencoder
train(autoencoder, data_loader, epochs=10, lr=1e-3)