import os
import sys
import logging

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def load_data(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    logging.info("Loading data...")
    training = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform
    )
    train_loader = DataLoader(training, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    logging.info("Data loaded!")
    return train_loader, test_loader


class MultinomialLogistic(nn.Module):
    def __init__(self):
        super(MultinomialLogistic, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return x


def fit_model(model, device, train_loader, learning_rate, momentum, epochs):
    logging.info("Fitting model...")
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
        logging.info(f"Epoch {epoch+1}: Loss = {running_loss}")
    logging.info("Model fit!")


def assess_fit(model, device, test):
    logging.info("Assessing model...")
    with torch.no_grad():
        total = 0
        total_correct = 0
        for data in test:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    logging.info(f"Accuracy: {total_correct/total}")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    MOMENTUM = float(os.getenv("MOMENTUM"))
    EPOCHS = int(os.getenv("EPOCHS"))

    train, test = load_data(BATCH_SIZE)
    model = MultinomialLogistic()
    model.to(device)
    fit_model(model, device, train, LEARNING_RATE, MOMENTUM, EPOCHS)
    assess_fit(model, device, test)


if __name__ == "__main__":
    main()
