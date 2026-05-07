import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generate import collate_fn
from models import RNNSeq2Seq, LSTMSeq2Seq

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_SIZE  = 32
HIDDEN_SIZE = 128
BATCH_SIZE  = 512
EPOCHS      = 50
LR          = 1e-3
TRAIN_SIZE  = 10_000
TEST_SIZE   = 2_000

# 词表覆盖 0-31 以支持分布外数值
MAX_VAL  = 31
PAD      = MAX_VAL + 1
SOS      = MAX_VAL + 2
VOCAB    = MAX_VAL + 3


class SortDataset(Dataset):
    def __init__(self, num_samples, min_len=5, max_len=8, min_val=0, max_val=15):
        self.samples = []
        nums = list(range(min_val, max_val + 1))
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            seq = random.sample(nums, length)
            self.samples.append((seq, sorted(seq, reverse=True)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    x_pad = [x + [PAD] * (max_len - len(x)) for x in xs]
    y_pad = [y + [PAD] * (max_len - len(y)) for y in ys]
    return (
        torch.tensor(x_pad, dtype=torch.long),
        torch.tensor(y_pad, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, tgt, lengths in loader:
            src, tgt, lengths = src.to(DEVICE), tgt.to(DEVICE), lengths.to(DEVICE)
            preds = model.predict(src, lengths, max_len=tgt.size(1))
            for i in range(src.size(0)):
                L = lengths[i].item()
                if preds[i, :L].tolist() == tgt[i, :L].tolist():
                    correct += 1
            total += src.size(0)
    return correct / total


def train_model(model_cls):
    train_loader = DataLoader(
        SortDataset(TRAIN_SIZE, min_len=5, max_len=8, min_val=0, max_val=15),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
    )
    model = model_cls(vocab_size=VOCAB, embed_size=EMBED_SIZE,
                      hidden_size=HIDDEN_SIZE, pad_idx=PAD, sos_idx=SOS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    for _ in range(EPOCHS):
        model.train()
        for src, tgt, lengths in train_loader:
            src, tgt, lengths = src.to(DEVICE), tgt.to(DEVICE), lengths.to(DEVICE)
            logits = model(src, tgt, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def make_loader(min_len, max_len, min_val, max_val):
    ds = SortDataset(TEST_SIZE, min_len=min_len, max_len=max_len,
                     min_val=min_val, max_val=max_val)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)


SCENARIOS = [
    ("in-distribution  (len 5-8,  val 0-15)",  dict(min_len=5,  max_len=8,  min_val=0,  max_val=15)),
    ("longer sequences (len 9-12, val 0-15)",  dict(min_len=9,  max_len=12, min_val=0,  max_val=15)),
    ("larger values    (len 5-8,  val 16-31)", dict(min_len=5,  max_len=8,  min_val=16, max_val=31)),
    ("both OOD         (len 9-12, val 16-31)", dict(min_len=9,  max_len=12, min_val=16, max_val=31)),
]


def evaluate_all(model, label, file=None):
    lines = [f"\n  {label}", f"  {'-'*54}"]
    for desc, kwargs in SCENARIOS:
        acc = accuracy(model, make_loader(**kwargs))
        lines.append(f"  {desc:<44}  acc={acc:.4f}")
    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    trained = {}
    for model_cls, name in [(RNNSeq2Seq, "Basic RNN"), (LSTMSeq2Seq, "LSTM")]:
        print(f"\nTraining {name} ...")
        trained[name] = train_model(model_cls)

    with open("results_q2.txt", "w") as f:
        f.write(f"Device: {DEVICE}\n")
        for name, model in trained.items():
            evaluate_all(model, name, file=f)

    print("\nSaved to results_q2.txt")
