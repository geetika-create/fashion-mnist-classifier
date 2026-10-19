"""Fashion-MNIST clothing classifier: a multi-layer perceptron built with PyTorch.

Trains a fully connected network on the 10-class Fashion-MNIST dataset,
plots the training loss, saves the model, and reports test accuracy."""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_PATH = "models/ann.pt"


# Data

def load_data(batch_size=BATCH_SIZE):
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_data, test_data, train_dataloader, test_dataloader


# Model

class MultiLayerPerceptron(nn.Module):
    """784 -> 256 -> 128 -> 32 -> 10 fully connected network with ReLU."""

    def __init__(self, input_features=784, output=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output),
        )

    def forward(self, x):
        return self.layers(x)


# Training

def train_loop(train_dataloader, model, loss_fn, optimizer, epochs):
    model.train()
    train_loss = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for image, label in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch: {epoch + 1} | Loss: {avg_loss:.4f}")

    return train_loss


def plot_loss(losses):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


# Evaluation

def accuracy(correct, total):
    return correct / total * 100


def test_loop(test_dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in tqdm(test_dataloader, desc="Testing"):
            pred = model(image)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            total += len(label)

    acc = accuracy(correct, total)
    print(f"Test accuracy: {acc:.2f}%")
    return acc


def predict_random_sample(model, test_data):
    model.eval()
    idx = torch.randint(0, len(test_data), (1,)).item()
    image, label = test_data[idx]

    with torch.no_grad():
        pred_idx = model(image.unsqueeze(0)).argmax(1).item()

    plt.figure(figsize=(5, 5))
    plt.title(
        f"Prediction: {test_data.classes[pred_idx]} | "
        f"Actual: {test_data.classes[label]}"
    )
    plt.imshow(image.squeeze(), cmap="gray")
    plt.show()


# Persistence

def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path=MODEL_PATH):
    model = MultiLayerPerceptron()
    model.load_state_dict(torch.load(path))
    return model


# Main

def main():
    train_data, test_data, train_dataloader, test_dataloader = load_data()
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")

    model = MultiLayerPerceptron()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    summary(model, input_size=(1, 28, 28))

    losses = train_loop(train_dataloader, model, loss_fn, optimizer, EPOCHS)
    plot_loss(losses)

    save_model(model)
    test_loop(test_dataloader, model)
    predict_random_sample(model, test_data)


if __name__ == "__main__":
    main()
