import os

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# 真实的高斯混合参数
true_pis = np.array([0.3, 0.4, 0.3])

true_mus = np.array([
    [0.0, 0.0],
    [3.0, 0.0],
    [2.0, 2.0]
])

true_sigmas = np.array([
    [[0.5, 0.0],
     [0.0, 0.8]],

    [[0.8, 0.3],
     [0.3, 0.5]],

    [[0.6, -0.2],
     [-0.2, 0.6]]
])

N = 1000
K = 3
D = 2

# 先采样成分标签，再从对应高斯分布采样
labels = np.random.choice(K, size=N, p=true_pis)
X = np.zeros((N, D))
for k in range(K):
    idx = labels == k
    X[idx] = np.random.multivariate_normal(
        mean=true_mus[k],
        cov=true_sigmas[k],
        size=np.sum(idx)
    )


def gaussian_pdf(X, mu, sigma):
    N, D = X.shape
    sigma = sigma + 1e-6 * np.eye(D)
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)

    diff = X - mu
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    coef = 1.0 / np.sqrt((2 * np.pi) ** D * det)
    return coef * np.exp(exponent)


def em_gmm(X, K=3, max_iter=100, tol=1e-5):
    N, D = X.shape

    # 均值初始化为随机样本点，协方差初始化为整体协方差
    pis = np.ones(K) / K
    init_idx = np.random.choice(N, K, replace=False)
    mus = X[init_idx].copy()
    cov_init = np.cov(X.T) + 1e-6 * np.eye(D)
    sigmas = np.array([cov_init.copy() for _ in range(K)])

    log_likelihoods = []

    for it in range(max_iter):
        # E-step: 责任度 gamma
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pis[k] * gaussian_pdf(X, mus[k], sigmas[k])
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-12)

        # M-step
        Nk = np.sum(gamma, axis=0)
        pis = Nk / N
        mus = gamma.T @ X / Nk[:, None]
        for k in range(K):
            diff = X - mus[k]
            sigmas[k] = (gamma[:, k][:, None] * diff).T @ diff / Nk[k]
            sigmas[k] += 1e-6 * np.eye(D)   # 防止协方差奇异

        likelihood = np.zeros((N, K))
        for k in range(K):
            likelihood[:, k] = pis[k] * gaussian_pdf(X, mus[k], sigmas[k])
        log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1) + 1e-12))
        log_likelihoods.append(log_likelihood)

        if it > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return pis, mus, sigmas, gamma, log_likelihoods


est_pis, est_mus, est_sigmas, gamma, log_likelihoods = em_gmm(X, K=3)
pred_labels = np.argmax(gamma, axis=1)


def match_components(true_mus, est_mus):
    """EM 估计出的成分顺序可能与真实成分置换，按均值最近距离做一一匹配。"""
    K = true_mus.shape[0]
    used = set()
    match = []
    for i in range(K):
        distances = np.linalg.norm(est_mus - true_mus[i], axis=1)
        for j in np.argsort(distances):
            if j not in used:
                match.append(j)
                used.add(j)
                break
    return np.array(match)


match = match_components(true_mus, est_mus)
est_pis_matched = est_pis[match]
est_mus_matched = est_mus[match]
est_sigmas_matched = est_sigmas[match]

np.set_printoptions(precision=4, suppress=True)

print("EM 迭代次数：", len(log_likelihoods))
print("最终对数似然：%.4f" % log_likelihoods[-1])

print("\n真实混合系数：")
print(true_pis)
print("估计混合系数：")
print(est_pis_matched)
print("混合系数误差：")
print(est_pis_matched - true_pis)

print("\n真实均值：")
print(true_mus)
print("估计均值：")
print(est_mus_matched)
print("均值误差（L2 范数）：")
print(np.linalg.norm(est_mus_matched - true_mus, axis=1))

print("\n真实协方差矩阵：")
print(true_sigmas)
print("估计协方差矩阵：")
print(est_sigmas_matched)
print("协方差矩阵误差（Frobenius 范数）：")
print(np.linalg.norm((est_sigmas_matched - true_sigmas).reshape(K, -1), axis=1))


x_min, x_max = -3, 6
y_min, y_max = -3, 5


def add_gaussian_contour(mu, sigma, ax, color, linestyle="-"):
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x, y)
    pos = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    Z = gaussian_pdf(pos, mu, sigma).reshape(X_grid.shape)
    ax.contour(X_grid, Y_grid, Z, levels=4, colors=color,
               linestyles=linestyle, linewidths=1.2)


fig, ax = plt.subplots(figsize=(8, 6.5))
ax.scatter(X[:, 0], X[:, 1], c=pred_labels, s=12, alpha=0.6, cmap="viridis")

# 实线=估计成分，虚线=真实成分
for k in range(K):
    add_gaussian_contour(est_mus[k], est_sigmas[k], ax, color="red", linestyle="-")
for k in range(K):
    add_gaussian_contour(true_mus[k], true_sigmas[k], ax, color="black", linestyle="--")

# contour 无法直接进图例，用代理线条标注
from matplotlib.lines import Line2D
contour_legend = [
    Line2D([0], [0], color="red", linestyle="-", label="estimated"),
    Line2D([0], [0], color="black", linestyle="--", label="true"),
]

ax.scatter(est_mus[:, 0], est_mus[:, 1], c="red", marker="x", s=120,
           linewidths=2.5, label="est. mean")
ax.scatter(true_mus[:, 0], true_mus[:, 1], c="black", marker="+", s=160,
           linewidths=2.5, label="true mean")

ax.set_title("EM clustering: estimated (solid) vs true (dashed) contours")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + contour_legend, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "em_clustering.png"), dpi=150)
plt.close(fig)


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(log_likelihoods, marker="o", ms=3)
ax.set_title("Log-likelihood during EM")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log-likelihood")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "em_loglikelihood.png"), dpi=150)
plt.close(fig)


# ---- 非参数密度估计 ----

grid_size = 120
grid_x = np.linspace(x_min, x_max, grid_size)
grid_y = np.linspace(y_min, y_max, grid_size)
GX, GY = np.meshgrid(grid_x, grid_y)
grid_points = np.column_stack([GX.ravel(), GY.ravel()])

# 真实密度，作为三种估计的对照
true_density = np.zeros(grid_points.shape[0])
for k in range(K):
    true_density += true_pis[k] * gaussian_pdf(grid_points, true_mus[k], true_sigmas[k])
true_density = true_density.reshape(GX.shape)

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(GX, GY, true_density, levels=30, cmap="viridis")
ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.25, c="white")
fig.colorbar(cf, ax=ax, label="density")
ax.set_title("True mixture density (reference)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "true_density.png"), dpi=150)
plt.close(fig)


# 直方图：改变 bin 数
bins_list = [10, 20, 40]
fig, axes = plt.subplots(1, len(bins_list), figsize=(5 * len(bins_list), 4.5))
for ax, bins in zip(axes, bins_list):
    h = ax.hist2d(
        X[:, 0], X[:, 1],
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
        density=True,
        cmap="viridis"
    )
    fig.colorbar(h[3], ax=ax, label="density")
    ax.set_title(f"Histogram, bins={bins}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
fig.suptitle("Histogram density estimation (varying bin width)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hist_density.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


def gaussian_kde_density(X, grid_points, h2):
    """各向同性高斯核密度估计，H^2 为带宽方差。"""
    N, D = X.shape
    # dist2[i, j] = ||grid_i - x_j||^2
    dist2 = (
        np.sum(grid_points ** 2, axis=1)[:, None]
        - 2 * grid_points @ X.T
        + np.sum(X ** 2, axis=1)[None, :]
    )
    kernel = np.exp(-dist2 / (2 * h2))
    return kernel.sum(axis=1) / (N * 2 * np.pi * h2)


# 高斯核：改变核方差 H^2
h2_list = [0.05, 0.2, 0.8]
fig, axes = plt.subplots(1, len(h2_list), figsize=(5 * len(h2_list), 4.5))
for ax, h2 in zip(axes, h2_list):
    Z = gaussian_kde_density(X, grid_points, h2).reshape(GX.shape)
    cf = ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, c="white")
    fig.colorbar(cf, ax=ax, label="density")
    ax.set_title(f"Gaussian KDE, $H^2={h2}$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
fig.suptitle("Gaussian kernel density estimation (varying $H^2$)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "kde_density.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


def knn_density(X, grid_points, K_neighbor):
    """二维 K 近邻密度估计： p(x) = K / (N * pi * r_K^2)。"""
    N, D = X.shape
    dist2 = (
        np.sum(grid_points ** 2, axis=1)[:, None]
        - 2 * grid_points @ X.T
        + np.sum(X ** 2, axis=1)[None, :]
    )
    dist2 = np.maximum(dist2, 0.0)
    rK2 = np.partition(dist2, K_neighbor - 1, axis=1)[:, K_neighbor - 1]
    volume = np.pi * rK2
    return K_neighbor / (N * volume + 1e-12)


# K 近邻：改变近邻数 K
K_list = [5, 20, 50]
fig, axes = plt.subplots(1, len(K_list), figsize=(5 * len(K_list), 4.5))
for ax, K_neighbor in zip(axes, K_list):
    Z = knn_density(X, grid_points, K_neighbor).reshape(GX.shape)
    cf = ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, c="white")
    fig.colorbar(cf, ax=ax, label="density")
    ax.set_title(f"KNN density, K={K_neighbor}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
fig.suptitle("K-nearest-neighbor density estimation (varying $K$)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "knn_density.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


print("\n所有图像已保存到：", FIG_DIR)
