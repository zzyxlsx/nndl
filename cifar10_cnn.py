"""
CIFAR-10 CNN 分类
- 卷积和池化使用 custom_layers.py 中手写实现
- 反向传播使用 PyTorch autograd
- 网络结构：9 个隐藏层（7 个卷积/池化 + 1 个 GlobalAvgPool + 1 个 FC）
"""

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


# ─────────────────────────── 网络架构 ───────────────────────────

class CIFAR10CNN(nn.Module):
    """
    9 个隐藏层的 CNN（不超过 10 层）：
      Block1: Conv(3→32) + Conv(32→32) + MaxPool  [隐藏层 1,2,3]
      Block2: Conv(32→64) + Conv(64→64) + MaxPool  [隐藏层 4,5,6]
      Block3: Conv(64→128) + GlobalAvgPool          [隐藏层 7,8]
      FC:     Linear(128→256)                       [隐藏层 9]
      Output: Linear(256→10)
    """

    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = ManualConv2d(3,  32, 3, padding=1)   # 隐藏层 1
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = ManualConv2d(32, 32, 3, padding=1)   # 隐藏层 2
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = ManualMaxPool2d(2, 2)                 # 隐藏层 3
        self.drop1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = ManualConv2d(32, 64, 3, padding=1)   # 隐藏层 4
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = ManualConv2d(64, 64, 3, padding=1)   # 隐藏层 5
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = ManualMaxPool2d(2, 2)                 # 隐藏层 6
        self.drop2 = nn.Dropout(0.25)

        # Block 3
        self.conv5 = ManualConv2d(64, 128, 3, padding=1)  # 隐藏层 7
        self.bn5   = nn.BatchNorm2d(128)
        self.gap   = ManualAvgPool2d(8, 8)                 # 隐藏层 8（Global Avg Pool）

        # Classifier
        self.fc1   = nn.Linear(128, 256)                   # 隐藏层 9
        self.drop3 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 10)                    # 输出层

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        # Block 2: 16x16 -> 8x8
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        # Block 3: 8x8 -> 1x1
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)                        # (N, 128, 1, 1)

        # Classifier
        x = x.view(x.size(0), -1)             # (N, 128)
        x = self.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


# ─────────────────────────── 数据预处理 ─────────────────────────

def build_loaders(batch_size=128):
    X_train_full, y_train_full, X_test, y_test = load_cifar10()

    val_size = 5000
    X_val,   y_val   = X_train_full[:val_size],  y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]

    # 归一化（per-channel mean/std）
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    def to_tensor_dataset(X, y):
        # reshape: (N, 3072) -> (N, 3, 32, 32)
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


# ─────────────────────────── 训练 / 评估 ────────────────────────

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 1 - correct / total   # error rate


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

        train_err = evaluate(model, train_loader)
        val_err   = evaluate(model, val_loader)
        history['train_err'].append(train_err)
        history['val_err'].append(val_err)
        print(f"Epoch {epoch:3d} | train_err={train_err:.4f} | val_err={val_err:.4f}")

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


# ─────────────────────────── 可视化 ─────────────────────────────

def plot_history(history, path='cifar10_cnn_curve.png'):
    epochs = range(1, len(history['train_err']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [1-e for e in history['train_err']], label='Train Acc')
    plt.plot(epochs, [1-e for e in history['val_err']],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('CIFAR-10 CNN Training Curve')
    plt.legend(); plt.tight_layout()
    plt.savefig(path); print(f'Saved {path}')


def plot_confusion(model, test_loader, path='cifar10_cnn_confusion.png'):
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


# ─────────────────────────── 主程序 ─────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')

    print('Building data loaders...')
    train_loader, val_loader, test_loader = build_loaders(batch_size=32)

    model = CIFAR10CNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')

    history = train(model, train_loader, val_loader,
                    max_epochs=30, patience=10, lr=1e-3, weight_decay=1e-4)

    # 测试集
    test_err = evaluate(model, test_loader)
    print(f'\nTest Accuracy: {(1-test_err)*100:.2f}%')

    torch.save(model.state_dict(), 'cifar10_cnn.pth')
    print('Model saved to cifar10_cnn.pth')

    plot_history(history)
    plot_confusion(model, test_loader)
