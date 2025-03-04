import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :encoded.size(1) // 2], encoded[:, encoded.size(1) // 2:]
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

# 2. Generative Adversarial Network (GAN)
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 3. Recurrent Neural Network (RNN) with Memory (LSTM)
class RNNMemory(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNMemory, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 4. Self-Diagnosis (Anomaly Detection using Reconstruction Error)
def self_diagnose(model, data_loader, threshold):
    anomalies = []
    for batch in data_loader:
        inputs = batch[0]
        reconstructed, _, _ = model(inputs)
        reconstruction_error = torch.mean((inputs - reconstructed) ** 2, dim=1)
        anomaly_indices = torch.where(reconstruction_error > threshold)[0]
        anomalies.extend(anomaly_indices.tolist())
    return anomalies

# Example Usage
if __name__ == "__main__":
    # Example Dataset (Replace with your actual data)
    input_dim = 10
    hidden_dim = 128
    latent_dim = 2
    output_dim = input_dim

    data = torch.randn(100, input_dim)
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=32)

    # VAE Training
    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer_vae = optim.Adam(vae.parameters(), lr=0.001)
    # ... (VAE Training Loop - calculate loss, backpropagate, optimize)

    # GAN Training
    gen = Generator(latent_dim, output_dim)
    disc = Discriminator(output_dim)
    optimizer_gen = optim.Adam(gen.parameters(), lr=0.001)
    optimizer_disc = optim.Adam(disc.parameters(), lr=0.001)
    # ... (GAN Training Loop - calculate loss, backpropagate, optimize)

    # RNN with Memory Training
    rnn = RNNMemory(input_dim, hidden_dim, output_dim)
    optimizer_rnn = optim.Adam(rnn.parameters(), lr=0.001)
    # ... (RNN Training Loop - calculate loss, backpropagate, optimize)

    # Self-Diagnosis
    threshold = 0.5  # Adjust threshold as needed
    anomalies = self_diagnose(vae, data_loader, threshold)
    print(f"Detected Anomalies: {anomalies}")

    # Example Dreaming (VAE)
    z_sample = torch.randn(1, latent_dim)
    dream = vae.decoder(z_sample)
    print("Dream:", dream)

    # Example Dreaming (GAN)
    z_gan_sample = torch.randn(1, latent_dim)
    dream_gan = gen(z_gan_sample)
    print("Dream GAN", dream_gan)

    #Example RNN prediction
    rnn_input = torch.randn(1,5,input_dim) #batch, sequence , input size
    rnn_prediction = rnn(rnn_input)
    print("RNN Prediction", rnn_prediction)
