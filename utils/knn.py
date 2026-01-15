import torch
import torch.nn.functional as F
from tqdm import tqdm

def build_latent_features(latents: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """
    latents: [N, 4, 32, 32]
    return:  [N, 4*pool*pool] L2-normalized features
    工业实践：不要直接 flatten 4096 维做 kNN，又慢又容易被噪声/近重复支配。
    """
    # float32 for stable similarity ranking
    x = latents.float()
    x = F.adaptive_avg_pool2d(x, (pool, pool))          # [N, 4, pool, pool]
    feat = x.flatten(1)                                 # [N, 4*pool*pool]
    feat = F.normalize(feat, dim=1, eps=1e-8)
    return feat


@torch.no_grad()
def precompute_knn_table(
    feat: torch.Tensor,
    k: int,
    chunk: int = 512,
    max_sim: float = 0.999,
    device_for_tables: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    feat: [N, D] normalized
    returns:
      knn_idx: [N, k] (long)
      knn_sim: [N, k] (float16)
    预计算每个样本的 top-k 邻居（排除 self；并过滤 sim > max_sim 的近重复，避免 seed amplification 锁死）
    """
    assert feat.dim() == 2
    N, D = feat.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to build kNN table.")
    k = int(min(k, N - 1))

    # Tables kept on chosen device for fast indexing during sampling
    table_device = feat.device if device_for_tables == "cuda" else torch.device("cpu")

    knn_idx = torch.empty((N, k), dtype=torch.long, device=table_device)
    knn_sim = torch.empty((N, k), dtype=torch.float16, device=table_device)

    feat_T = feat.t()  # [D, N]

    # We'll compute sims in chunks to avoid huge memory spikes
    for start in tqdm(range(0, N, chunk), desc=f"Precomputing kNN (N={N}, D={D}, k={k})", leave=False):
        end = min(N, start + chunk)
        # [chunk, N]
        sims = feat[start:end] @ feat_T

        # exclude self
        rows = torch.arange(end - start, device=feat.device)
        cols = rows + start
        sims[rows, cols] = -1e9

        # filter near-duplicates (very high cosine similarity)
        if max_sim is not None:
            sims = sims.masked_fill(sims > float(max_sim), -1e9)

        vals, inds = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)  # [chunk,k]

        if table_device.type == "cpu":
            knn_idx[start:end] = inds.to("cpu", non_blocking=True)
            knn_sim[start:end] = vals.to(torch.float16).to("cpu", non_blocking=True)
        else:
            knn_idx[start:end] = inds
            knn_sim[start:end] = vals.to(torch.float16)

    return knn_idx, knn_sim


@torch.no_grad()
def sample_knn_pairs(
    N: int,
    knn_idx: torch.Tensor,
    knn_sim: torch.Tensor,
    batch_size: int,
    device: torch.device,
    temperature: float = 0.2,
    min_sim: float = -1.0,
    mutual: bool = False,
    max_tries: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 idx1, idx2，都在同一个 class bucket 内。
    - idx1 随机
    - idx2 从 idx1 的 top-k 邻居里按 softmax(sim/temperature) 采样
    - 支持 mutual-kNN 约束（更严格）
    """
    idx1 = torch.randint(0, N, (batch_size,), device=device)

    # gather neighbors and sims (ensure on same device)
    if knn_idx.device != device:
        knn_idx_d = knn_idx.to(device, non_blocking=True)
        knn_sim_d = knn_sim.to(device, non_blocking=True).float()
    else:
        knn_idx_d = knn_idx
        knn_sim_d = knn_sim.float()

    neigh = knn_idx_d[idx1]   # [B,k]
    sims = knn_sim_d[idx1]    # [B,k]

    # fallback mask if no good neighbors
    best = sims.max(dim=1).values
    bad = best < float(min_sim)

    # sample neighbor position
    if temperature is None or temperature <= 0:
        j = torch.randint(0, neigh.shape[1], (batch_size,), device=device)
    else:
        probs = torch.softmax(sims / float(temperature), dim=1)
        j = torch.multinomial(probs, 1).squeeze(1)

    idx2 = neigh[torch.arange(batch_size, device=device), j]

    # fallback for bad rows
    if bad.any():
        idx2[bad] = torch.randint(0, N, (int(bad.sum().item()),), device=device)

    if not mutual:
        return idx1, idx2

    # mutual-kNN: require idx1 in knn(idx2)
    for _ in range(max_tries):
        neigh2 = knn_idx_d[idx2]  # [B,k]
        ok = (neigh2 == idx1.unsqueeze(1)).any(dim=1)
        if ok.all():
            break
        # resample idx2 where not ok
        not_ok = ~ok
        sims_bad = knn_sim_d[idx1[not_ok]]
        neigh_bad = knn_idx_d[idx1[not_ok]]
        if temperature is None or temperature <= 0:
            j_bad = torch.randint(0, neigh_bad.shape[1], (neigh_bad.shape[0],), device=device)
        else:
            probs_bad = torch.softmax(sims_bad / float(temperature), dim=1)
            j_bad = torch.multinomial(probs_bad, 1).squeeze(1)
        idx2_new = neigh_bad[torch.arange(neigh_bad.shape[0], device=device), j_bad]
        idx2[not_ok] = idx2_new

    return idx1, idx2