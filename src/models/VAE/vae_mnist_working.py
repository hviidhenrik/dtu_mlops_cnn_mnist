"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image

# Model Hyperparameters
dataset_path = "datasets"
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
x_dim = 784

# hyperparams
batch_size = 100
hidden_dim = 80
latent_dim = 40
lr = 1e-3
epochs = 10

wandb_config = dict(
    batch_size=batch_size,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    lr=lr,
    epochs=epochs,
)

wandb.init(
    project="dtu_mlops-vae_mnist_working", notes="just a trial run", config=wandb_config
)

# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

train_dataset = TensorDataset(
    train_dataset.data.type(torch.float32) / 255, train_dataset.targets
)
test_dataset = TensorDataset(
    test_dataset.data.type(torch.float32) / 255, test_dataset.targets
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)
        return z, mean, log_var

    @staticmethod
    def reparameterization(
        mean,
        std,
    ):
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
wandb.watch(model, log_freq=100)

BCE_loss = nn.BCELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()
    epoch_avg_loss = overall_loss / (batch_idx * batch_size)
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", epoch_avg_loss)

    # Generate reconstruction fromn test set after every epoch
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

        # Generate samples
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    input_images = make_grid(x.view(batch_size, 1, 28, 28))
    reconstructed_images = make_grid(x_hat.view(batch_size, 1, 28, 28))
    generated_images = make_grid(generated_images.view(batch_size, 1, 28, 28))

    wandb.log(
        {
            "epoch_avg_loss": epoch_avg_loss,
            "input_images": wandb.Image(input_images),
            "reconstructed_images": wandb.Image(reconstructed_images),
            "generated_images": wandb.Image(generated_images),
        }
    )
print("Finish!!")

# Generate reconstructions
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        x_hat, _, _ = model(x)
        break

save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

# Generate samples
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")
