"""
第 3 题：自编码器与自监督学习（CIFAR-10）

比较三种表示学习方法：自编码器（ae）、旋转角度预测（rot）、掩码自编码器（mae）。
三者共用同一个小型 ViT 编码器，以保证架构、参数量、表示维度一致。该编码器支持变长
输入（可见 token 通过位置索引 gather 对应的位置编码），因此既能编码全部 64 个块，
也能只编码 MAE 遮蔽后剩下的 16 个块。下游表示统一取编码器输出 token 的 mean-pool。

用法：
  python q3.py --task ae   --device cuda:0
  python q3.py --task rot  --device cuda:1
  python q3.py --task mae  --device cuda:2
  python q3.py --task probe --device cuda:0     # 需在以上三个训练完成后运行
"""

import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS_DIR, "outputs")
FIG_DIR = os.path.join(_THIS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 复用 hw1 已下载的 CIFAR-10
CIFAR_DIR = os.path.join(_THIS_DIR, "..", "..", "hw1", "data", "cifar-10-batches-py")

IMG_SIZE = 32
PATCH = 4
GRID = IMG_SIZE // PATCH          # 8
NUM_PATCHES = GRID * GRID         # 64
PATCH_DIM = PATCH * PATCH * 3     # 48
EMBED_DIM = 192
DEPTH = 6
HEADS = 3

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].astype(np.float32)
    y = np.array(d[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10():
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    X_test, y_test = _load_batch(os.path.join(CIFAR_DIR, "test_batch"))

    def to_tensor(X):
        X = X.reshape(-1, 3, 32, 32) / 255.0
        X = (X - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
        return torch.from_numpy(X.astype(np.float32))

    return (to_tensor(X_train), torch.from_numpy(y_train),
            to_tensor(X_test), torch.from_numpy(y_test))


def denormalize(img):
    """标准化空间 -> [0,1]，用于可视化。"""
    mean = torch.tensor(CIFAR_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def patchify(imgs):
    """(B,3,32,32) -> (B,64,48)，块内按 (C,p,p) 展平。"""
    B = imgs.shape[0]
    x = imgs.reshape(B, 3, GRID, PATCH, GRID, PATCH)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, NUM_PATCHES, PATCH_DIM)


def unpatchify(patches):
    """(B,64,48) -> (B,3,32,32)。"""
    B = patches.shape[0]
    x = patches.reshape(B, GRID, GRID, 3, PATCH, PATCH)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 3, IMG_SIZE, IMG_SIZE)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, depth=DEPTH, heads=HEADS):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, patches, ids=None):
        # ids: (B,M) 每个 token 在 0..63 中的位置；None 表示输入全部 64 个块
        x = self.patch_embed(patches)
        B, M, Dd = x.shape
        pos = self.pos_embed.expand(B, -1, -1)
        if ids is not None:
            pos = torch.gather(pos, 1, ids.unsqueeze(-1).expand(-1, -1, Dd))
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def encode_pool(self, imgs):
        """编码整张图并 mean-pool，得到下游用的 192 维表示。"""
        tokens = self.forward(patchify(imgs), ids=None)
        return tokens.mean(dim=1)


class MaskedAutoencoder(nn.Module):
    """统一的（掩码）自编码器：AE 用 mask_ratio=0、对全部块算重构 MSE；
    MAE 用 mask_ratio=0.75、仅对被遮蔽块算 MSE。解码器为轻量 ViT。"""

    def __init__(self, dec_dim=128, dec_depth=2, dec_heads=4):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder_embed = nn.Linear(EMBED_DIM, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, dec_dim))
        self.dec_blocks = nn.ModuleList(
            [Block(dec_dim, dec_heads) for _ in range(dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, PATCH_DIM)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

    @staticmethod
    def random_masking(patches, mask_ratio, generator=None):
        """随机排序 64 个块并保留前 len_keep 个。mask 中 1=被遮蔽。"""
        B, Np, _ = patches.shape
        len_keep = int(round(Np * (1 - mask_ratio)))
        noise = torch.rand(B, Np, device=patches.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        visible = torch.gather(
            patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )
        mask = torch.ones(B, Np, device=patches.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask, ids_restore, ids_keep

    def forward_decoder(self, latent, ids_restore):
        # 把可见 token 和 mask_token 拼回 64 个，再按 ids_restore 还原原始顺序
        x = self.decoder_embed(latent)
        B, M, Dd = x.shape
        mask_tokens = self.mask_token.expand(B, NUM_PATCHES - M, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, Dd))
        x = x + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        return self.dec_pred(x)

    def forward(self, imgs, mask_ratio, masked_only, generator=None):
        patches = patchify(imgs)
        visible, mask, ids_restore, ids_keep = self.random_masking(
            patches, mask_ratio, generator
        )
        latent = self.encoder(visible, ids=ids_keep)
        pred = self.forward_decoder(latent, ids_restore)

        per_patch = ((pred - patches) ** 2).mean(dim=-1)
        if masked_only:
            loss = (per_patch * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = per_patch.mean()
        return loss, pred, mask


class RotationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTEncoder()
        self.head = nn.Linear(EMBED_DIM, 4)

    def forward(self, imgs):
        return self.head(self.encoder.encode_pool(imgs))


def rotate_batch(imgs, ks):
    """对每张图按 ks[i] (0..3) 旋转 90*k 度。"""
    out = imgs.clone()
    for k in (1, 2, 3):
        m = ks == k
        if m.any():
            out[m] = torch.rot90(imgs[m], k, dims=[-2, -1])
    return out


def iterate_minibatches(n, batch_size, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_reconstruction(task, X_train, X_test, device, epochs, batch_size, lr):
    mask_ratio = 0.0 if task == "ae" else 0.75
    masked_only = (task == "mae")

    model = MaskedAutoencoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = X_train.shape[0]   # 数据留在 CPU，按 batch 搬 GPU（省显存）
    for ep in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for batch_idx in iterate_minibatches(n, batch_size, shuffle=True):
            imgs = X_train[batch_idx].to(device)
            loss, _, _ = model(imgs, mask_ratio, masked_only)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{task}] epoch {ep+1}/{epochs}  train_loss={running/nb:.4f}",
                  flush=True)

    test_mse = evaluate_reconstruction(model, X_test, device, mask_ratio,
                                       masked_only, batch_size)
    print(f"[{task}] 测试集重构 MSE（标准化空间，"
          f"{'仅遮蔽块' if masked_only else '全部块'}）= {test_mse:.4f}", flush=True)

    torch.save(model.encoder.state_dict(), os.path.join(OUT_DIR, f"encoder_{task}.pt"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_{task}.pt"))

    if task == "ae":
        plot_ae_reconstruction(model, X_test, device)
    else:
        plot_mae_reconstruction(model, X_test, device)

    with open(os.path.join(OUT_DIR, f"{task}_result.txt"), "w") as f:
        f.write(f"test_recon_mse={test_mse:.6f}\n")
    return test_mse


@torch.no_grad()
def evaluate_reconstruction(model, X_test, device, mask_ratio, masked_only,
                            batch_size):
    # 固定随机种子产生一次遮蔽，保证测试 MSE 可复现
    model.eval()
    n = X_test.shape[0]
    gen = torch.Generator(device=device).manual_seed(123)
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        imgs = X_test[i:i + batch_size].to(device)
        loss, _, _ = model(imgs, mask_ratio, masked_only, generator=gen)
        bs = imgs.shape[0]
        total += loss.item() * bs
        count += bs
    return total / count


@torch.no_grad()
def plot_ae_reconstruction(model, X_test, device, n_show=8):
    model.eval()
    imgs = X_test[:n_show].to(device)
    _, pred, _ = model(imgs, 0.0, False)
    orig = denormalize(imgs).cpu()
    recon = denormalize(unpatchify(pred)).cpu()

    fig, axes = plt.subplots(2, n_show, figsize=(1.4 * n_show, 3))
    for j in range(n_show):
        axes[0, j].imshow(orig[j].permute(1, 2, 0).numpy())
        axes[0, j].axis("off")
        axes[1, j].imshow(recon[j].permute(1, 2, 0).numpy())
        axes[1, j].axis("off")
    fig.text(0.02, 0.72, "original", va="center")
    fig.text(0.02, 0.28, "recon", va="center")
    fig.suptitle("Auto-Encoder reconstruction (test set)")
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "ae_reconstruction.png"), dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_mae_reconstruction(model, X_test, device, n_show=8):
    model.eval()
    imgs = X_test[:n_show].to(device)
    gen = torch.Generator(device=device).manual_seed(7)
    _, pred, mask = model(imgs, 0.75, True, generator=gen)
    patches = patchify(imgs)

    m = mask.unsqueeze(-1)
    masked_patches = patches * (1 - m)               # 遮蔽块置 0
    pasted = patches * (1 - m) + pred * m            # 可见块用原图，遮蔽块用预测

    orig = denormalize(imgs).cpu()
    masked_img = denormalize(unpatchify(masked_patches)).cpu()
    recon_img = denormalize(unpatchify(pasted)).cpu()

    fig, axes = plt.subplots(3, n_show, figsize=(1.4 * n_show, 4.4))
    for j in range(n_show):
        axes[0, j].imshow(orig[j].permute(1, 2, 0).numpy()); axes[0, j].axis("off")
        axes[1, j].imshow(masked_img[j].permute(1, 2, 0).numpy()); axes[1, j].axis("off")
        axes[2, j].imshow(recon_img[j].permute(1, 2, 0).numpy()); axes[2, j].axis("off")
    fig.text(0.02, 0.80, "original", va="center")
    fig.text(0.02, 0.50, "masked", va="center")
    fig.text(0.02, 0.18, "recon", va="center")
    fig.suptitle("MAE: original / masked (75%) / reconstruction (test set)")
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "mae_reconstruction.png"), dpi=150)
    plt.close(fig)


def train_rotation(X_train, X_test, device, epochs, batch_size, lr):
    model = RotationModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = X_train.shape[0]
    for ep in range(epochs):
        model.train()
        running, correct, seen = 0.0, 0, 0
        for batch_idx in iterate_minibatches(n, batch_size, shuffle=True):
            imgs = X_train[batch_idx].to(device)
            ks = torch.randint(0, 4, (imgs.shape[0],), device=device)
            logits = model(rotate_batch(imgs, ks))
            loss = F.cross_entropy(logits, ks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            correct += (logits.argmax(1) == ks).sum().item()
            seen += imgs.shape[0]
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[rot] epoch {ep+1}/{epochs}  loss={running:.1f} "
                  f"train_acc={correct/seen:.4f}", flush=True)

    test_acc = evaluate_rotation(model, X_test, device, batch_size)
    print(f"[rot] 测试集旋转预测准确率 = {test_acc:.4f}", flush=True)

    torch.save(model.encoder.state_dict(), os.path.join(OUT_DIR, "encoder_rot.pt"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_rot.pt"))
    with open(os.path.join(OUT_DIR, "rot_result.txt"), "w") as f:
        f.write(f"test_rot_acc={test_acc:.6f}\n")
    return test_acc


@torch.no_grad()
def evaluate_rotation(model, X_test, device, batch_size):
    # 对每张测试图的 4 个旋转都预测，统计总体准确率
    model.eval()
    n = X_test.shape[0]
    correct, total = 0, 0
    for k in range(4):
        ks = torch.full((batch_size,), k, device=device)
        for i in range(0, n, batch_size):
            imgs = X_test[i:i + batch_size].to(device)
            kk = ks[:imgs.shape[0]]
            logits = model(rotate_batch(imgs, kk))
            correct += (logits.argmax(1) == kk).sum().item()
            total += imgs.shape[0]
    return correct / total


@torch.no_grad()
def extract_features(encoder, X, device, batch_size=512):
    encoder.eval()
    n = X.shape[0]
    feats = []
    for i in range(0, n, batch_size):
        imgs = X[i:i + batch_size].to(device)
        feats.append(encoder.encode_pool(imgs).cpu())
    return torch.cat(feats, dim=0)


def train_linear_probe(feat_train, y_train, feat_test, y_test, device,
                       epochs=60, lr=1e-3, batch_size=256):
    """在冻结的编码器特征上训练线性分类器，返回测试准确率。"""
    in_dim = feat_train.shape[1]
    clf = nn.Linear(in_dim, 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 标准化特征，稳定线性分类训练
    mu = feat_train.mean(0, keepdim=True)
    sd = feat_train.std(0, keepdim=True) + 1e-6
    Ftr = ((feat_train - mu) / sd).to(device)
    Fte = ((feat_test - mu) / sd).to(device)
    ytr = y_train.to(device)
    yte = y_test.to(device)

    n = Ftr.shape[0]
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            loss = F.cross_entropy(clf(Ftr[bi]), ytr[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        return (clf(Fte).argmax(1) == yte).float().mean().item()


def run_probe(device, epochs, batch_size):
    from sklearn.manifold import TSNE

    set_seed(0)
    X_train, y_train, X_test, y_test = load_cifar10()

    methods = ["ae", "rot", "mae"]
    titles = {"ae": "Auto-Encoder", "rot": "Rotation", "mae": "MAE"}
    accs = {}
    test_feats = {}

    for m in methods:
        ckpt = os.path.join(OUT_DIR, f"encoder_{m}.pt")
        if not os.path.exists(ckpt):
            print(f"[probe] 缺少 {ckpt}，跳过 {m}", flush=True)
            continue
        enc = ViTEncoder().to(device)
        enc.load_state_dict(torch.load(ckpt, map_location=device))

        f_tr = extract_features(enc, X_train, device)
        f_te = extract_features(enc, X_test, device)
        acc = train_linear_probe(f_tr, y_train, f_te, y_test, device, epochs=epochs)
        accs[m] = acc
        test_feats[m] = f_te
        print(f"[probe] {titles[m]:12s} 线性分类测试准确率 = {acc:.4f}", flush=True)

    with open(os.path.join(OUT_DIR, "probe_result.txt"), "w") as f:
        for m in methods:
            if m in accs:
                f.write(f"{m}_linear_acc={accs[m]:.6f}\n")

    # t-SNE 取测试集子集可视化
    n_vis = 3000
    sub = np.random.RandomState(0).choice(X_test.shape[0], n_vis, replace=False)
    y_sub = y_test.numpy()[sub]

    valid = [m for m in methods if m in test_feats]
    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5.5))
    if len(valid) == 1:
        axes = [axes]
    for ax, m in zip(axes, valid):
        feats = test_feats[m].numpy()[sub]
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=0).fit_transform(feats)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sub, cmap="tab10", s=6, alpha=0.6)
        ax.set_title(f"{titles[m]} (acc={accs[m]:.3f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, fraction=0.02, label="class")
    fig.suptitle("t-SNE of frozen encoder features (CIFAR-10 test)", y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "tsne_features.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[probe] t-SNE 图已保存", flush=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["ae", "rot", "mae", "probe"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    if args.task == "probe":
        epochs = args.epochs if args.epochs is not None else 60
        run_probe(device, epochs, args.batch_size)
        return

    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"train={tuple(X_train.shape)} test={tuple(X_test.shape)} device={device}",
          flush=True)
    print(f"共享编码器参数量 = {count_params(ViTEncoder()):,}  表示维度 = {EMBED_DIM}",
          flush=True)

    if args.task in ("ae", "mae"):
        epochs = args.epochs if args.epochs is not None else 40
        train_reconstruction(args.task, X_train, X_test, device, epochs,
                             args.batch_size, args.lr)
    elif args.task == "rot":
        epochs = args.epochs if args.epochs is not None else 30
        train_rotation(X_train, X_test, device, epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
