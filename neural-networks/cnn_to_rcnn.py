import torch

from torch import nn
from torch.optim import Adam


class CNN(nn.Module):
    def __init__(self, size: int) -> None:
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(2500, size)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x

    def train_phase(self, epoch: int, optimizer: Adam, criterion: nn.CrossEntropyLoss, X_train, y_train, X_test, y_test) -> None:
        train_losses, val_losses = list(), list()

        if torch.cuda.is_available():
            model = self.cuda()
            criterion = criterion.cuda()

        self.train()
        tr_loss = 0

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        optimizer.zero_grad()

        output_train = self(X_train)
        output_val = self(X_test)

        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_test)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()

        if epoch % 2 == 0:
            print(f'Epoch: {epoch + 1}\tLoss: {loss_val}')
            print(f'Train Loss: {tr_loss}')
