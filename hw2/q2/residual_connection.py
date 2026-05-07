import torch

torch.manual_seed(0)

n = 20
d = 8

# 环形图邻接矩阵
A = torch.zeros(n, n)
for i in range(n):
    A[i, (i + 1) % n] = 1
    A[(i + 1) % n, i] = 1

I = torch.eye(n)
A_tilde = A + I

D_tilde = torch.diag(A_tilde.sum(dim=1))
D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D_tilde)))
S = D_inv_sqrt @ A_tilde @ D_inv_sqrt

H0 = torch.randn(n, d)

def mean_cosine_similarity(H):
    H_norm = H / (H.norm(dim=1, keepdim=True) + 1e-8)
    sim = H_norm @ H_norm.T
    mask = ~torch.eye(H.shape[0], dtype=torch.bool)
    return sim[mask].mean().item()

def vanilla_gcn(S, H0, K):
    H = H0.clone()
    for _ in range(K):
        H = S @ H
    return H

def residual_gcn(S, H0, K, alpha=0.2):
    # H^{k+1} = (1-alpha) S H^k + alpha H0
    H = H0.clone()
    for _ in range(K):
        H = (1 - alpha) * (S @ H) + alpha * H0
    return H

print("K\t原始GCN余弦相似度\t残差GCN余弦相似度")
for K in [0, 1, 2, 5, 10, 20, 50, 100]:
    H_vanilla = vanilla_gcn(S, H0, K)
    H_res = residual_gcn(S, H0, K, alpha=0.2)

    sim_vanilla = mean_cosine_similarity(H_vanilla)
    sim_res = mean_cosine_similarity(H_res)

    print(f"{K:3d}\t{sim_vanilla:.6f}\t\t{sim_res:.6f}")
