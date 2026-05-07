import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False)
        z_hat = (z - mean) / torch.sqrt(var + self.eps)
        return self.gamma * z_hat + self.beta


class LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, keepdim=True, unbiased=False)
        z_hat = (z - mean) / torch.sqrt(var + self.eps)
        return self.gamma * z_hat + self.beta


class MLP(nn.Module):
    def __init__(self, input_feature_size: int, hidden_feature_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_feature_size, hidden_feature_size)
        self.bn1 = BatchNorm(hidden_feature_size)

        self.fc2 = nn.Linear(hidden_feature_size, hidden_feature_size)
        self.ln2 = LayerNorm(hidden_feature_size)

        self.fc3 = nn.Linear(hidden_feature_size, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        a0 = (x - mean) / (std + 1e-5)

        z1 = self.fc1(a0)
        z1_bn = self.bn1(z1)
        a1 = self.relu(z1_bn)

        z2 = self.fc2(a1)
        z2_ln = self.ln2(z2)
        a2 = self.relu(z2_ln)

        out = self.fc3(a2)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    input_feature_size = 16
    hidden_feature_size = 32

    model = MLP(input_feature_size, hidden_feature_size)
    x = torch.randn(batch_size, input_feature_size)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, 1), "Output shape mismatch"
    print("Shape check passed.")
