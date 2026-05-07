import numpy as np
import pickle
import os
import urllib.request
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neural_network import FeedForwardNet

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.join(_SRC_DIR, '..')
_OUT_DIR   = os.path.join(_ROOT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)

CIFAR_URL  = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_DIR  = os.path.join(_ROOT_DIR, 'data', 'cifar-10-batches-py')
CIFAR_TAR  = os.path.join(_ROOT_DIR, 'data', 'cifar-10-python.tar.gz')

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def download_cifar10():
    if not os.path.exists(CIFAR_DIR):
        if not os.path.exists(CIFAR_TAR):
            print('Downloading CIFAR-10...')
            urllib.request.urlretrieve(CIFAR_URL, CIFAR_TAR)
        print('Extracting...')
        with tarfile.open(CIFAR_TAR, 'r:gz') as t:
            t.extractall(os.path.join(_ROOT_DIR, 'data'))
        print('Done.')

def load_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    X = d[b'data'].astype(np.float32)
    y = np.array(d[b'labels'])
    return X, y

def load_cifar10():
    download_cifar10()
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(CIFAR_DIR, f'data_batch_{i}'))
        Xs.append(X); ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_batch(os.path.join(CIFAR_DIR, 'test_batch'))
    return X_train, y_train, X_test, y_test


def preprocess(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def plot_history(history, save_path=None):
    if save_path is None:
        save_path = os.path.join(_OUT_DIR, 'cifar10_training_curve.png')
    epochs = range(1, len(history['train_err']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [1 - e for e in history['train_err']], label='Train Acc')
    plt.plot(epochs, [1 - e for e in history['val_err']],   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CIFAR-10 MLP Training (Pure SGD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Curve saved to {save_path}')

def plot_confusion(y_true, y_pred, save_path=None):
    if save_path is None:
        save_path = os.path.join(_OUT_DIR, 'cifar10_confusion.png')
    C = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap='Blues')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Test Set)')
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(C[i, j]), ha='center', va='center',
                    color='white' if C[i, j] > C.max() * 0.5 else 'black', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Confusion matrix saved to {save_path}')

def print_per_class_acc(y_true, y_pred):
    print('\n── Per-class Accuracy ──')
    for c, name in enumerate(CIFAR10_CLASSES):
        mask = y_true == c
        acc  = np.mean(y_pred[mask] == c)
        print(f'  {name:12s}: {acc*100:.1f}%')


if __name__ == '__main__':
    np.random.seed(42)

    print('Loading CIFAR-10...')
    X_train_full, y_train_full, X_test, y_test = load_cifar10()

    val_size = 5000
    X_val,   y_val   = X_train_full[:val_size],  y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:],  y_train_full[val_size:]

    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    net = FeedForwardNet(
        layer_sizes=[3072, 512, 256, 10],
        hidden_activation='relu',
        lr=0.1,
        lam=1e-4,
    )

    net.fit(X_train, y_train, X_val, y_val,
            max_epochs=30, batch_size=256, shuffle_seed=0, verbose=True)

    y_pred    = net.predict(X_test)
    test_acc  = np.mean(y_pred == y_test)
    print(f'\nTest Accuracy: {test_acc*100:.2f}%')

    print_per_class_acc(y_test, y_pred)

    plot_history(net.history)
    plot_confusion(y_test, y_pred)
