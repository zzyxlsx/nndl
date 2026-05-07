import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from custom_layers import ManualConv2d, ManualMaxPool2d, ManualAvgPool2d
from cifar10_mlp import load_cifar10, CIFAR10_CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, '..')
_OUT_DIR  = os.path.join(_ROOT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)


class CIFAR10CNN(nn.Module):
    """
    9 hidden layers:
      Block1: Conv(3->32) + Conv(32->32) + MaxPool
      Block2: Conv(32->64) + Conv(64->64) + MaxPool
      Block3: Conv(64->128) + GlobalAvgPool
      FC:     Linear(128->256)
      Output: Linear(256->10)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = ManualConv2d(3,  32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = ManualConv2d(32, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = ManualMaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = ManualConv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = ManualConv2d(64, 64, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = ManualMaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = ManualConv2d(64, 128, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.gap   = ManualAvgPool2d(8, 8)

        self.fc1   = nn.Linear(128, 256)
        self.drop3 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 10)

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


def build_loaders(batch_size=128):
    X_train_full, y_train_full, X_test, y_test = load_cifar10()

    val_size = 5000
    X_val,   y_val   = X_train_full[:val_size],  y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    def to_tensor_dataset(X, y):
        Xt = torch.tensor(X, dtype=torch.float32).view(-1, 3, 32, 32)
        yt = torch.tensor(y, dtype=torch.long)
        return TensorDataset(Xt, yt)

    train_loader = DataLoader(to_tensor_dataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(to_tensor_dataset(X_val,   y_val),
                              batch_size=256,       shuffle=False, num_workers=0)
    test_loader  = DataLoader(to_tensor_dataset(X_test,  y_test),
                              batch_size=256,       shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 1 - correct / total


def train(model, train_loader, val_loader,
          max_epochs=50, patience=10, lr=1e-3, weight_decay=1e-4):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    history    = {'train_err': [], 'val_err': []}
    best_val   = float('inf')
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_err = evaluate(model, val_loader)
        if epoch % 5 == 0:
            train_err = evaluate(model, train_loader)
        else:
            train_err = float('nan')
        history['train_err'].append(train_err)
        history['val_err'].append(val_err)
        train_str = f"{train_err:.4f}" if not np.isnan(train_err) else "  --  "
        print(f"Epoch {epoch:3d} | train_err={train_str} | val_err={val_err:.4f}")

        if val_err < best_val - 1e-5:
            best_val   = val_err
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history


def plot_history(history, path=None):
    if path is None:
        path = os.path.join(_OUT_DIR, 'cifar10_cnn_curve.png')
    epochs = range(1, len(history['train_err']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [1-e for e in history['train_err']], label='Train Acc')
    plt.plot(epochs, [1-e for e in history['val_err']],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('CIFAR-10 CNN Training Curve')
    plt.legend(); plt.tight_layout()
    plt.savefig(path); print(f'Saved {path}')


def plot_confusion(model, test_loader, path=None):
    if path is None:
        path = os.path.join(_OUT_DIR, 'cifar10_cnn_confusion.png')
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            all_pred.append(model(X.to(DEVICE)).argmax(1).cpu())
            all_true.append(y)
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    C = np.zeros((10, 10), int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap='Blues')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('CNN Confusion Matrix')
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(C[i, j]), ha='center', va='center',
                    color='white' if C[i, j] > C.max()*0.5 else 'black', fontsize=7)
    plt.tight_layout(); plt.savefig(path); print(f'Saved {path}')

    per_class = C.diagonal() / C.sum(axis=1)
    print('\n── Per-class Accuracy ──')
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f'  {name:12s}: {per_class[i]*100:.1f}%')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')

    print('Building data loaders...')
    train_loader, val_loader, test_loader = build_loaders(batch_size=128)

    model = CIFAR10CNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')

    history = train(model, train_loader, val_loader,
                    max_epochs=30, patience=10, lr=1e-3, weight_decay=1e-4)

    test_err = evaluate(model, test_loader)
    print(f'\nTest Accuracy: {(1-test_err)*100:.2f}%')

    model_path = os.path.join(_OUT_DIR, 'cifar10_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    plot_history(history)
    plot_confusion(model, test_loader)
