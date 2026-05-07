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

H = torch.randn(n, d)

def mean_cosine_similarity(H):
    H_norm = H / H.norm(dim=1, keepdim=True)
    sim = H_norm @ H_norm.T
    mask = ~torch.eye(H.shape[0], dtype=torch.bool)
    return sim[mask].mean().item()

for K in [0, 1, 2, 5, 10, 20, 50, 100]:
    H_K = H.clone()
    for _ in range(K):
        H_K = S @ H_K

    sim = mean_cosine_similarity(H_K)
    print(f"K = {K:3d}, 平均余弦相似度 = {sim:.6f}")
