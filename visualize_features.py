"""
特征映射可视化
- 卷积核可视化（第一层）
- 各隐藏层 feature map 可视化（forward hook）
- 不同类别图像的激活对比
需先运行 cifar10_cnn.py 生成 cifar10_cnn.pth
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifar10_cnn import CIFAR10CNN, build_loaders, DEVICE
from custom_layers import ManualConv2d
from cifar10_mlp import CIFAR10_CLASSES


# ─────────────────────────── 加载模型 ───────────────────────────

def load_model():
    model = CIFAR10CNN().to(DEVICE)
    model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=DEVICE))
    model.eval()
    return model


# ─────────────────────────── 1. 卷积核可视化 ────────────────────

def plot_conv_kernels(model, path='vis_kernels.png'):
    """可视化第一层 32 个 3x3x3 卷积核（RGB 彩色）"""
    w = model.conv1.weight.detach().cpu()  # (32, 3, 3, 3)
    # 归一化到 [0,1]
    w_min, w_max = w.min(), w.max()
    w = (w - w_min) / (w_max - w_min + 1e-8)

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle('Conv1 Kernels (32 filters, 3×3×3)', fontsize=13)
    for i, ax in enumerate(axes.flat):
        kernel = w[i].permute(1, 2, 0).numpy()  # (3,3,3) -> (3,3,3) HWC
        ax.imshow(kernel)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f'Saved {path}')


# ─────────────────────────── 2. Feature Map 可视化 ──────────────

def register_hooks(model):
    """在每个 ManualConv2d 后注册 forward hook，捕获激活输出"""
    feature_maps = {}
    hooks = []

    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    for name in layer_names:
        module = getattr(model, name)
        def make_hook(n):
            def hook(mod, inp, out):
                feature_maps[n] = out.detach().cpu()
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    return feature_maps, hooks


def plot_feature_maps(model, image_tensor, label, path_prefix='vis_fmap'):
    """
    对单张图像跑 forward，可视化每层前 16 个 channel 的 feature map。
    image_tensor: (1, 3, 32, 32)
    """
    feature_maps, hooks = register_hooks(model)

    with torch.no_grad():
        model(image_tensor.to(DEVICE))

    for h in hooks:
        h.remove()

    layer_titles = {
        'conv1': 'Layer 1 – Conv(3→32)  32×32',
        'conv2': 'Layer 2 – Conv(32→32) 32×32',
        'conv3': 'Layer 4 – Conv(32→64) 16×16',
        'conv4': 'Layer 5 – Conv(64→64) 16×16',
        'conv5': 'Layer 7 – Conv(64→128) 8×8',
    }

    for name, fmap in feature_maps.items():
        # fmap: (1, C, H, W)
        fmap = fmap[0]          # (C, H, W)
        n_show = min(16, fmap.shape[0])
        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        fig.suptitle(f'{layer_titles[name]}  |  input: {label}', fontsize=11)
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                fm = fmap[i].numpy()
                ax.imshow(fm, cmap='viridis')
                ax.set_title(f'ch{i}', fontsize=7)
            ax.axis('off')
        plt.tight_layout()
        save_path = f'{path_prefix}_{name}_{label}.png'
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f'Saved {save_path}')


# ─────────────────────────── 3. 不同类别激活对比 ────────────────

def plot_activation_comparison(model, test_loader, path='vis_compare.png'):
    """
    从测试集各取一张正确分类的图，可视化 conv1 激活的均值强度对比。
    """
    model.eval()
    samples = {}   # class_id -> image tensor

    for X, y in test_loader:
        with torch.no_grad():
            preds = model(X.to(DEVICE)).argmax(1).cpu()
        for i in range(len(y)):
            c = y[i].item()
            if c not in samples and preds[i].item() == c:
                samples[c] = X[i]
        if len(samples) == 10:
            break

    feature_maps, hooks = register_hooks(model)
    mean_acts = {name: [] for name in ['conv1', 'conv3', 'conv5']}

    for c in range(10):
        img = samples[c].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            model(img)
        for name in mean_acts:
            mean_acts[name].append(feature_maps[name][0].mean().item())

    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Mean Activation per Class at Different Layers', fontsize=12)
    for ax, (name, vals) in zip(axes, mean_acts.items()):
        ax.bar(CIFAR10_CLASSES, vals, color='steelblue')
        ax.set_title(name)
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Activation')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f'Saved {path}')


# ─────────────────────────── 主程序 ─────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    model = load_model()
    print('Model loaded.')

    _, _, test_loader = build_loaders(batch_size=256)

    # 1. 卷积核
    plot_conv_kernels(model)

    # 2. 对 3 个类别各取一张图，可视化各层 feature map
    samples = {}
    for X, y in test_loader:
        with torch.no_grad():
            preds = model(X.to(DEVICE)).argmax(1).cpu()
        for i in range(len(y)):
            c = y[i].item()
            if c not in samples and preds[i].item() == c and c in [0, 3, 8]:
                # airplane(0), cat(3), ship(8)
                samples[c] = (X[i], CIFAR10_CLASSES[c])
        if len(samples) == 3:
            break

    for c, (img, label) in samples.items():
        plot_feature_maps(model, img.unsqueeze(0), label)

    # 3. 各类别激活对比
    plot_activation_comparison(model, test_loader)

    print('\nAll visualizations done.')
