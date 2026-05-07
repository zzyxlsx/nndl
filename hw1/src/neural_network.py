import numpy as np


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
    if z.ndim == 1:
        e = np.exp(z - np.max(z))
        return e / e.sum()
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'relu':    (relu,    relu_prime),
}


def cross_entropy_loss(y_hat, y_onehot):
    return -np.sum(y_onehot * np.log(np.clip(y_hat, 1e-12, 1.0)))


class FeedForwardNet:
    def __init__(self, layer_sizes, hidden_activation='relu', lr=0.01, lam=1e-4):
        self.L = len(layer_sizes) - 1
        self.lr = lr
        self.lam = lam
        self.act, self.act_prime = ACTIVATIONS[hidden_activation]

        self.W = []
        self.b = []
        for l in range(self.L):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            self.W.append(np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in) * 0.1)
            self.b.append(np.zeros(fan_out))

    def _feedforward(self, x):
        a = [x]
        z_list = []

        for l in range(self.L):
            z = a[l] @ self.W[l].T + self.b[l]
            z_list.append(z)
            if l < self.L - 1:
                a.append(self.act(z))
            else:
                a.append(softmax(z))

        return z_list, a

    def _backprop(self, z_list, a, y_onehot):
        delta = [None] * self.L
        dW    = [None] * self.L
        db    = [None] * self.L

        delta[-1] = a[-1] - y_onehot

        for l in range(self.L - 2, -1, -1):
            delta[l] = (delta[l + 1] @ self.W[l + 1]) * self.act_prime(z_list[l])

        for l in range(self.L):
            dW[l] = delta[l].T @ a[l] / len(a[0])
            db[l] = delta[l].mean(axis=0)

        return dW, db

    def _update(self, dW, db):
        for l in range(self.L):
            self.W[l] -= self.lr * (dW[l] + self.lam * self.W[l])
            self.b[l] -= self.lr * db[l]

    def predict(self, X):
        _, a = self._feedforward(X)
        return np.argmax(a[-1], axis=1)

    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)

    def fit(self, X_train, y_train, X_val, y_val,
            max_epochs=30, batch_size=256, shuffle_seed=0, verbose=True):
        N = len(y_train)
        n_classes = self.W[-1].shape[0]
        self.history = {'train_err': [], 'val_err': []}

        for epoch in range(1, max_epochs + 1):
            idx = np.random.RandomState(shuffle_seed + epoch).permutation(N)

            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                X_b = X_train[batch_idx]
                y_oh = np.zeros((len(batch_idx), n_classes))
                y_oh[np.arange(len(batch_idx)), y_train[batch_idx]] = 1.0

                z_list, a = self._feedforward(X_b)
                dW, db    = self._backprop(z_list, a, y_oh)
                self._update(dW, db)

            val_err   = self.error_rate(X_val, y_val)
            train_err = self.error_rate(X_train, y_train) if epoch % 5 == 0 else float('nan')
            self.history['train_err'].append(train_err)
            self.history['val_err'].append(val_err)

            if verbose:
                train_str = f"{train_err:.4f}" if not np.isnan(train_err) else "  --  "
                print(f"Epoch {epoch:4d} | train_err={train_str} | val_err={val_err:.4f}", flush=True)

        return self


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
