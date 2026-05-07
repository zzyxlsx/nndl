import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifar10_mlp import (load_cifar10, preprocess, plot_history,
                          plot_confusion, print_per_class_acc, CIFAR10_CLASSES)
from neural_network import FeedForwardNet

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, '..')
_OUT_DIR  = os.path.join(_ROOT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)


class TorchMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def copy_weights_from_numpy(torch_model, numpy_net):
    linear_layers = [m for m in torch_model.net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        layer.weight.data = torch.tensor(numpy_net.W[i], dtype=torch.float32)
        layer.bias.data   = torch.tensor(numpy_net.b[i], dtype=torch.float32)


def train_torch(torch_model, X_train, y_train, X_val, y_val,
                lr, lam, max_epochs=30, batch_size=256, shuffle_seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model = torch_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=lr, weight_decay=lam)

    N = len(y_train)
    history = {'train_err': [], 'val_err': []}

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32).to(device)

    for epoch in range(1, max_epochs + 1):
        torch_model.train()
        idx = np.random.RandomState(shuffle_seed + epoch).permutation(N)

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X_tr_t[batch_idx].to(device)
            yb = y_tr_t[batch_idx].to(device)

            optimizer.zero_grad()
            loss = criterion(torch_model(xb), yb)
            loss.backward()
            optimizer.step()

        torch_model.eval()
        with torch.no_grad():
            val_pred  = torch_model(X_val_t).argmax(dim=1).cpu().numpy()
            val_err   = np.mean(val_pred != y_val)

            if epoch % 5 == 0:
                train_pred = torch_model(X_tr_t.to(device)).argmax(dim=1).cpu().numpy()
                train_err  = np.mean(train_pred != y_train)
                print(f"Epoch {epoch:4d} | train_err={train_err:.4f} | val_err={val_err:.4f}", flush=True)
            else:
                train_err = float('nan')
                print(f"Epoch {epoch:4d} | train_err=  --   | val_err={val_err:.4f}", flush=True)

        history['train_err'].append(train_err)
        history['val_err'].append(val_err)

    return history


if __name__ == '__main__':
    SEED        = 42
    LAYER_SIZES = [3072, 512, 256, 10]
    LR          = 0.1
    LAM         = 1e-4
    MAX_EPOCHS  = 30
    PATIENCE    = 8
    BATCH_SIZE  = 256

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print('Loading CIFAR-10...')
    X_train_full, y_train_full, X_test, y_test = load_cifar10()
    val_size = 5000
    X_val,   y_val   = X_train_full[:val_size],  y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    np.random.seed(SEED)
    numpy_net = FeedForwardNet(LAYER_SIZES, hidden_activation='relu', lr=LR, lam=LAM)

    torch_model = TorchMLP(LAYER_SIZES)
    copy_weights_from_numpy(torch_model, numpy_net)

    history = train_torch(torch_model, X_train, y_train, X_val, y_val,
                          lr=LR, lam=LAM, max_epochs=MAX_EPOCHS,
                          batch_size=BATCH_SIZE, shuffle_seed=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = torch_model(X_te_t).argmax(dim=1).cpu().numpy()
    test_acc = np.mean(y_pred == y_test)
    print(f'\nTest Accuracy: {test_acc*100:.2f}%')
    print_per_class_acc(y_test, y_pred)

    plot_history(history,  save_path=os.path.join(_OUT_DIR, 'cifar10_torch_curve.png'))
    plot_confusion(y_test, y_pred, save_path=os.path.join(_OUT_DIR, 'cifar10_torch_confusion.png'))
