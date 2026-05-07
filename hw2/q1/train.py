import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate import SortDataset, collate_fn, PAD
from models import RNNSeq2Seq, LSTMSeq2Seq

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_SIZE   = 32
HIDDEN_SIZE  = 128
BATCH_SIZE   = 512
EPOCHS       = 50
LR           = 1e-3
TRAIN_SIZE   = 10_000
TEST_SIZE    = 2_000


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, tgt, lengths in loader:
            src, tgt, lengths = src.to(device), tgt.to(device), lengths.to(device)
            preds = model.predict(src, lengths, max_len=tgt.size(1))
            for i in range(src.size(0)):
                L = lengths[i].item()
                if preds[i, :L].tolist() == tgt[i, :L].tolist():
                    correct += 1
            total += src.size(0)
    return correct / total


def train_model(model_cls, name):
    print(f"\n{'='*60}")
    print(f"  Training {name}")
    print(f"{'='*60}")

    train_ds = SortDataset(TRAIN_SIZE)
    test_ds  = SortDataset(TEST_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    model = model_cls(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for src, tgt, lengths in train_loader:
            src, tgt, lengths = src.to(DEVICE), tgt.to(DEVICE), lengths.to(DEVICE)
            logits = model(src, tgt, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            acc = accuracy(model, test_loader, DEVICE)
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

    final_acc = accuracy(model, test_loader, DEVICE)
    best_acc  = max(best_acc, final_acc)
    print(f"\n  {name}  final test accuracy: {final_acc:.4f}  (best: {best_acc:.4f})")
    return final_acc


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    rnn_acc  = train_model(RNNSeq2Seq,  "Basic RNN Seq2Seq")
    lstm_acc = train_model(LSTMSeq2Seq, "LSTM Seq2Seq")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Basic RNN  test accuracy: {rnn_acc:.4f}")
    print(f"  LSTM       test accuracy: {lstm_acc:.4f}")
    print(f"  LSTM improvement over RNN: {lstm_acc - rnn_acc:+.4f}")
