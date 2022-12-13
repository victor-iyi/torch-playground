import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(
    root: str = 'data/',
    batch_size: int = 64,
    shuffle: bool = True,
    train: bool = True,
    download: bool = True
) -> DataLoader:

    dataset = datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=ToTensor()
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer
) -> None:
    size = len(data_loader.dataset)  # type: ignore
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module
) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    size = len(data_loader.dataset)  # type: ignore
    num_batches = len(data_loader)
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print('Test Error:')
    print(f'Acc: {correct:>0.2%}')
    print(f'Avg loss: {test_loss/num_batches:>8f}\n')


def main() -> int:
    # Load data.
    train_data = load_data(train=True)
    test_data = load_data(train=False)

    # Build model.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork().to(device)
    print(model)

    # Loss function & optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training / Testing loop.
    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')

        train(train_data, model, loss_fn, optimizer)
        test(test_data, model, loss_fn)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
