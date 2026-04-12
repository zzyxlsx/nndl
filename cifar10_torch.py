"""
CIFAR-10 分类：PyTorch 自动微分版本
网络结构、初始化、超参数与 cifar10_mlp.py 完全一致，仅反向传播改用 autograd。
用于验证手写反向传播实现的正确性。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用数据加载和预处理
from cifar10_mlp import (load_cifar10, preprocess, plot_history,
                          plot_confusion, print_per_class_acc, CIFAR10_CLASSES)
from neural_network import FeedForwardNet


# ─────────────────────────── PyTorch 网络 ───────────────────────

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
        return self.net(x)   # 输出 logits，CrossEntropyLoss 内含 softmax


def copy_weights_from_numpy(torch_model, numpy_net):
    """将 numpy 版网络的权重复制到 PyTorch 模型，保证初始化完全一致。"""
    linear_layers = [m for m in torch_model.net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        layer.weight.data = torch.tensor(numpy_net.W[i], dtype=torch.float32)
        layer.bias.data   = torch.tensor(numpy_net.b[i], dtype=torch.float32)


# ─────────────────────────── 训练（纯 SGD，逐样本）─────────────

def train_torch(torch_model, X_train, y_train, X_val, y_val,
                lr, lam, max_epochs=30, patience=8):
    criterion = nn.CrossEntropyLoss()
    # SGD + L2 正则（weight_decay 对应 λ）
    optimizer = optim.SGD(torch_model.parameters(), lr=lr, weight_decay=lam)

    N = len(y_train)
    history = {'train_err': [], 'val_err': []}
    best_val_err = float('inf')
    no_improve   = 0
    best_state   = None

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t = torch.tensor(y_val,   dtype=torch.long)

    for epoch in range(1, max_epochs + 1):
        torch_model.train()
        idx = np.random.permutation(N)

        for n in idx:
            x  = X_tr_t[n].unsqueeze(0)   # (1, D)
            yt = y_tr_t[n].unsqueeze(0)   # (1,)

            optimizer.zero_grad()
            logits = torch_model(x)
            loss   = criterion(logits, yt)
            loss.backward()
            optimizer.step()

        # 评估
        torch_model.eval()
        with torch.no_grad():
            val_pred   = torch_model(X_val_t).argmax(dim=1).numpy()
            val_err    = np.mean(val_pred != y_val)

            if epoch % 5 == 0:
                train_pred = torch_model(X_tr_t).argmax(dim=1).numpy()
                train_err  = np.mean(train_pred != y_train)
                print(f"Epoch {epoch:4d} | train_err={train_err:.4f} | val_err={val_err:.4f}")
            else:
                train_err = float('nan')
                print(f"Epoch {epoch:4d} | train_err=  --   | val_err={val_err:.4f}")

        history['train_err'].append(train_err)
        history['val_err'].append(val_err)

        if val_err < best_val_err - 1e-5:
            best_val_err = val_err
            no_improve   = 0
            best_state   = {k: v.clone() for k, v in torch_model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val_err={best_val_err:.4f})")
                break

    if best_state:
        torch_model.load_state_dict(best_state)
    return history


# ─────────────────────────── 主程序 ─────────────────────────────

if __name__ == '__main__':
    SEED        = 42
    LAYER_SIZES = [3072, 512, 256, 10]
    LR          = 0.001
    LAM         = 1e-4
    MAX_EPOCHS  = 30
    PATIENCE    = 8

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. 加载数据（与 numpy 版完全相同的划分）
    print('Loading CIFAR-10...')
    X_train_full, y_train_full, X_test, y_test = load_cifar10()
    val_size = 5000
    X_val,   y_val   = X_train_full[:val_size],  y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    # 2. 先构建 numpy 版网络，只为获得相同的初始权重
    np.random.seed(SEED)
    numpy_net = FeedForwardNet(LAYER_SIZES, hidden_activation='relu', lr=LR, lam=LAM)

    # 3. 构建 PyTorch 网络，复制相同初始权重
    torch_model = TorchMLP(LAYER_SIZES)
    copy_weights_from_numpy(torch_model, numpy_net)

    # 4. 训练
    history = train_torch(torch_model, X_train, y_train, X_val, y_val,
                          lr=LR, lam=LAM, max_epochs=MAX_EPOCHS, patience=PATIENCE)

    # 5. 测试集评估
    torch_model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, dtype=torch.float32)
        y_pred = torch_model(X_te_t).argmax(dim=1).numpy()
    test_acc = np.mean(y_pred == y_test)
    print(f'\nTest Accuracy: {test_acc*100:.2f}%')
    print_per_class_acc(y_test, y_pred)

    # 6. 可视化
    plot_history(history,  save_path='cifar10_torch_curve.png')
    plot_confusion(y_test, y_pred, save_path='cifar10_torch_confusion.png')
