"""
Q4: 在鞍点损失 L(θ1, θ2) = θ1² - θ2² 上对比 SGD、RMSprop、AdaDelta、Adam。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def loss(theta):
    return theta[0] ** 2 - theta[1] ** 2

def grad(theta):
    return np.array([2 * theta[0], -2 * theta[1]])


def run_sgd(theta0, alpha, n_steps):
    theta = theta0.copy()
    path = [theta.copy()]
    for _ in range(n_steps):
        theta = theta - alpha * grad(theta)
        path.append(theta.copy())
    return np.array(path)


def run_rmsprop(theta0, alpha, n_steps, rho=0.9, eps=1e-8):
    theta = theta0.copy()
    G = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(n_steps):
        g = grad(theta)
        G = rho * G + (1 - rho) * g ** 2
        theta = theta - alpha / np.sqrt(G + eps) * g
        path.append(theta.copy())
    return np.array(path)


def run_adadelta(theta0, alpha, n_steps, rho=0.9, eps=1e-8):
    theta = theta0.copy()
    G    = np.zeros_like(theta)
    dX2  = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(n_steps):
        g   = grad(theta)
        G   = rho * G + (1 - rho) * g ** 2
        dx  = -np.sqrt(dX2 + eps) / np.sqrt(G + eps) * g
        theta = theta + alpha * dx
        dX2 = rho * dX2 + (1 - rho) * dx ** 2
        path.append(theta.copy())
    return np.array(path)


def run_adam(theta0, alpha, n_steps, beta1=0.9, beta2=0.999, eps=1e-8):
    theta = theta0.copy()
    M = np.zeros_like(theta)
    G = np.zeros_like(theta)
    path = [theta.copy()]
    for t in range(1, n_steps + 1):
        g = grad(theta)
        M = beta1 * M + (1 - beta1) * g
        G = beta2 * G + (1 - beta2) * g ** 2
        M_hat = M / (1 - beta1 ** t)
        G_hat = G / (1 - beta2 ** t)
        theta = theta - alpha * M_hat / (np.sqrt(G_hat) + eps)
        path.append(theta.copy())
    return np.array(path)


np.random.seed(42)
theta0 = np.array([0.1, 0.1])
alpha  = 0.05
N      = 300

paths = {
    "SGD":      run_sgd(theta0, alpha, N),
    "RMSprop":  run_rmsprop(theta0, alpha, N),
    "AdaDelta": run_adadelta(theta0, alpha, N),
    "Adam":     run_adam(theta0, alpha, N),
}

colors = {
    "SGD":      "#e41a1c",
    "RMSprop":  "#377eb8",
    "AdaDelta": "#4daf4a",
    "Adam":     "#ff7f00",
}

# 图 1：损失曲面上的优化轨迹
t1 = np.linspace(-3, 3, 400)
t2 = np.linspace(-3, 3, 400)
T1, T2 = np.meshgrid(t1, t2)
Z = T1 ** 2 - T2 ** 2

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, (name, path) in zip(axes, paths.items()):
    cf = ax.contourf(T1, T2, Z, levels=40, cmap="RdBu_r", alpha=0.75)
    ax.contour(T1, T2, Z, levels=20, colors="gray", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, shrink=0.82, label="L(θ)")

    mask = (np.abs(path[:, 0]) <= 3) & (np.abs(path[:, 1]) <= 3)
    disp = path[mask]
    idx  = np.where(mask)[0]

    if len(disp) > 1:
        ax.plot(disp[:, 0], disp[:, 1], color=colors[name], lw=1.0, alpha=0.6)
    sc = ax.scatter(disp[:, 0], disp[:, 1],
                    c=idx, cmap="plasma", s=12, zorder=4, linewidths=0)
    plt.colorbar(sc, ax=ax, label="iteration", shrink=0.82)

    ax.plot(*theta0, "k^", ms=9, zorder=5, label=f"start {tuple(theta0)}")
    ax.plot(0, 0, "w*", ms=11, zorder=5, label="saddle (0,0)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$\theta_1$", fontsize=11)
    ax.set_ylabel(r"$\theta_2$", fontsize=11)
    n_shown = len(disp)
    ax.set_title(
        f"{name}  —  {n_shown}/{N+1} pts in view",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle(
    r"Optimization trajectories on $L(\theta_1,\theta_2)=\theta_1^2-\theta_2^2$"
    f"\n$\\alpha={alpha}$,  start={tuple(theta0)},  {N} steps",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trajectories.png")

# 图 2：|θ2| 随迭代变化，反映逃离鞍点的速度
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax_left = axes2[0]
for name, path in paths.items():
    vals = np.abs(path[:101, 1])
    ax_left.plot(vals, label=name, color=colors[name], lw=2.0)
ax_left.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="|θ₂|=1 threshold")
ax_left.set_xlabel("Iteration", fontsize=11)
ax_left.set_ylabel(r"$|\theta_2|$", fontsize=11)
ax_left.set_title("Early escape dynamics (first 100 steps)", fontsize=12)
ax_left.set_yscale("log")
ax_left.legend(fontsize=10)
ax_left.grid(True, which="both", alpha=0.3)

ax_right = axes2[1]
for name, path in paths.items():
    vals = np.abs(path[:, 1])
    # SGD 数值迅速发散，裁剪到 30 以便可视化
    clipped = np.clip(vals, None, 30)
    ls = "--" if name == "SGD" else "-"
    ax_right.plot(clipped, label=name + (" (clipped at 30)" if name == "SGD" else ""),
                  color=colors[name], lw=2.0, ls=ls)
ax_right.set_xlabel("Iteration", fontsize=11)
ax_right.set_ylabel(r"$|\theta_2|$  (clipped at 30)", fontsize=11)
ax_right.set_title("Full 300 steps (linear scale)", fontsize=12)
ax_right.legend(fontsize=10)
ax_right.grid(True, alpha=0.3)

fig2.suptitle(
    r"Escape speed from saddle point along $\theta_2$  ($\alpha=0.05$)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("escape_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved escape_speed.png")


THRESH = 1.0
print(f"\nEscape threshold |θ2| > {THRESH}")
print(f"{'Optimizer':<12} {'Escape iter':>12}  {'|θ2| at step 300':>18}")
for name, path in paths.items():
    escaped = np.where(np.abs(path[:, 1]) > THRESH)[0]
    esc_iter = int(escaped[0]) if len(escaped) else "never"
    print(f"{name:<12} {str(esc_iter):>12}  {abs(path[-1, 1]):>18.4f}")
