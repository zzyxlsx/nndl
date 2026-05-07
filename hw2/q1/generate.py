import random
import torch
from torch.utils.data import Dataset, DataLoader

PAD = 16
SOS = 17
EOS = 18
VOCAB_SIZE = 19

class SortDataset(Dataset):
    def __init__(self, num_samples, min_len=5, max_len=8, min_val=0, max_val=15):
        self.samples = []
        nums = list(range(min_val, max_val + 1))

        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            seq = random.sample(nums, length)
            target = sorted(seq, reverse=True)
            self.samples.append((seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)

    x_pad = []
    y_pad = []

    for x, y in zip(xs, ys):
        x_pad.append(x + [PAD] * (max_len - len(x)))
        y_pad.append(y + [PAD] * (max_len - len(y)))

    return (
        torch.tensor(x_pad, dtype=torch.long),
        torch.tensor(y_pad, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )