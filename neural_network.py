import numpy as np


# ─────────────────────────── 激活函数 ───────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'relu':    (relu,    relu_prime),
}


# ─────────────────────────── 损失函数 ───────────────────────────

def cross_entropy_loss(y_hat, y_onehot):
    return -np.sum(y_onehot * np.log(np.clip(y_hat, 1e-12, 1.0)))


# ─────────────────────────── 网络主体 ───────────────────────────

class FeedForwardNet:
    """
    多层感知机（前馈神经网络），支持多分类。

    Parameters
    ----------
    layer_sizes : list[int]
        每层神经元数，含输入层和输出层。
        例如 [784, 128, 64, 10] 表示输入784维，两个隐层，输出10类。
    hidden_activation : str
        隐层激活函数，'sigmoid' 或 'relu'。
    lr : float
        学习率 α。
    lam : float
        L2 正则化系数 λ。
    """

    def __init__(self, layer_sizes, hidden_activation='relu', lr=0.01, lam=1e-4):
        self.L = len(layer_sizes) - 1          # 层数（不含输入层）
        self.lr = lr
        self.lam = lam
        self.act, self.act_prime = ACTIVATIONS[hidden_activation]

        # 随机初始化权重（He 初始化）和偏置
        self.W = []
        self.b = []
        for l in range(self.L):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            self.W.append(np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in) * 0.1)
            self.b.append(np.zeros(fan_out))

    # ── 前馈 ──────────────────────────────────────────────────────

    def _feedforward(self, x):
        """
        返回每层的净输入 z 和激活值 a（a[0] = 输入 x）。
        输出层使用 softmax，隐层使用 self.act。
        """
        a = [x]          # a[0] = x（输入层激活值）
        z_list = []      # z[l] 对应第 l+1 层（1-indexed）

        for l in range(self.L):
            z = self.W[l] @ a[l] + self.b[l]
            z_list.append(z)
            if l < self.L - 1:
                a.append(self.act(z))
            else:
                a.append(softmax(z))   # 输出层

        return z_list, a

    # ── 反向传播 ──────────────────────────────────────────────────

    def _backprop(self, z_list, a, y_onehot):
        """
        计算每层误差项 δ，返回 dW, db 列表。
        输出层：δ = a_out - y（softmax + 交叉熵的组合梯度）
        隐层：δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ f'(z^(l))
        """
        delta = [None] * self.L
        dW    = [None] * self.L
        db    = [None] * self.L

        # 输出层误差（softmax + cross-entropy 的解析梯度）
        delta[-1] = a[-1] - y_onehot

        # 隐层误差（从倒数第二层向前）
        for l in range(self.L - 2, -1, -1):
            delta[l] = (self.W[l + 1].T @ delta[l + 1]) * self.act_prime(z_list[l])

        # 计算梯度
        for l in range(self.L):
            dW[l] = np.outer(delta[l], a[l])   # δ^(l) · (a^(l-1))^T
            db[l] = delta[l]

        return dW, db

    # ── 参数更新 ──────────────────────────────────────────────────

    def _update(self, dW, db):
        for l in range(self.L):
            self.W[l] -= self.lr * (dW[l] + self.lam * self.W[l])
            self.b[l] -= self.lr * db[l]

    # ── 预测 ──────────────────────────────────────────────────────

    def predict(self, X):
        """X: (N, D)，返回预测类别数组 (N,)"""
        preds = []
        for x in X:
            _, a = self._feedforward(x)
            preds.append(np.argmax(a[-1]))
        return np.array(preds)

    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)

    # ── SGD 训练（算法 4.1）──────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val,
            max_epochs=200, patience=10, verbose=True):
        """
        Parameters
        ----------
        X_train, y_train : 训练集特征 (N,D) 和标签 (N,)（整数类别）
        X_val,   y_val   : 验证集
        max_epochs       : 最大迭代轮数
        patience         : 早停耐心值（验证错误率连续 patience 轮不下降则停止）
        """
        N = len(y_train)
        n_classes = self.W[-1].shape[0]

        best_val_err = np.inf
        no_improve   = 0
        self.history = {'train_err': [], 'val_err': []}

        for epoch in range(1, max_epochs + 1):
            # 随机打乱训练集
            idx = np.random.permutation(N)

            for n in idx:
                x      = X_train[n]
                y_oh   = np.zeros(n_classes)
                y_oh[y_train[n]] = 1.0

                # 前馈
                z_list, a = self._feedforward(x)
                # 反向传播 + 梯度计算
                dW, db = self._backprop(z_list, a, y_oh)
                # 参数更新
                self._update(dW, db)

            # 验证集错误率（早停判断）
            val_err   = self.error_rate(X_val, y_val)
            train_err = self.error_rate(X_train, y_train) if epoch % 5 == 0 else float('nan')
            self.history['train_err'].append(train_err)
            self.history['val_err'].append(val_err)

            if verbose:
                train_str = f"{train_err:.4f}" if not np.isnan(train_err) else "  --  "
                print(f"Epoch {epoch:4d} | train_err={train_str} | val_err={val_err:.4f}")

            if val_err < best_val_err - 1e-5:
                best_val_err = val_err
                no_improve   = 0
                self._best_W = [w.copy() for w in self.W]
                self._best_b = [b.copy() for b in self.b]
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val_err={best_val_err:.4f})")
                    break

        # 恢复最优参数
        if hasattr(self, '_best_W'):
            self.W, self.b = self._best_W, self._best_b

        return self


# ─────────────────────────── 简单演示 ───────────────────────────

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X, y = data.data, data.target

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te)

    net = FeedForwardNet(
        layer_sizes=[4, 16, 16, 3],
        hidden_activation='relu',
        lr=0.01,
        lam=1e-4,
    )
    net.fit(X_tr, y_tr, X_val, y_val, max_epochs=300, patience=20)

    print(f"\nTest error rate: {net.error_rate(X_te, y_te):.4f}")
