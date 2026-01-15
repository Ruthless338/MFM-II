# scripts/generate_aug_fewshot_noise.py
'''
ISIC
------------------------------
Original Train size: 8012
Few-shot (10%) size: 801
Unused size:         7211
------------------------------
Class distribution in Few-shot set:
  MEL: 89
  NV: 536
  BCC: 41
  AKIEC: 26
  BKL: 88
  DF: 9
  VASC: 12
------------------------------
将每个类别都扩充到 Target Count = max Class_num 张，得到 Balance Dataset

本脚本：纯噪声采样 baseline
- 不再使用 inversion / 类内配对 / 插值
- 直接从 N(0,1) 噪声按类别采样，再通过 FM ODE 生成 latent，VAE decode 成图
'''
import argparse
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import DiT
from data.dataset import ISICLatentDataset

# -----------------------------------------------------------------------------
# Flow Matching ODE Solver
# -----------------------------------------------------------------------------

class FlowODESolver:
    """
    Flow Matching 的核心：ODE 求解器
    - Inversion: t=1 -> 0（本脚本不使用）
    - Sampling : t=0 -> 1（可使用 CFG）
    数值积分：Heun（二阶预测-校正）
    """
    def __init__(self, model, num_steps=50, device="cuda"):
        self.model = model
        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def _predict_v(self, x, t_scalar, y, cfg_scale: float = 1.0):
        """
        预测速度场 v(x,t|y)
        CFG 形式：
            v = v_uncond + s * (v_cond - v_uncond)
        其中 uncond 的 label 使用 null token id = num_classes
        """
        t = torch.ones(x.shape[0], device=self.device) * float(t_scalar)

        # 不启用 CFG：直接条件预测
        if cfg_scale is None or float(cfg_scale) == 1.0:
            return self.model(x, t, y)

        null_id = getattr(self.model, "num_classes", None)
        if null_id is None:
            raise AttributeError("Model must have attribute `num_classes` for CFG null label id.")

        y_null = torch.full_like(y, fill_value=null_id)

        v_uncond = self.model(x, t, y_null)
        v_cond = self.model(x, t, y)

        s = float(cfg_scale)
        return v_uncond + s * (v_cond - v_uncond)

    @torch.no_grad()
    def step(self, x, t, y, dt, cfg_scale: float = 1.0):
        """
        Heun (Improved Euler / predictor-corrector):
          v0 = v(x_t, t)
          x_euler = x_t + v0 * dt
          v1 = v(x_euler, t+dt)
          x_{t+dt} = x_t + 0.5*(v0+v1)*dt
        """
        v0 = self._predict_v(x, t, y, cfg_scale=cfg_scale)
        x_euler = x + v0 * dt
        v1 = self._predict_v(x_euler, t + dt, y, cfg_scale=cfg_scale)
        return x + 0.5 * (v0 + v1) * dt

    @torch.no_grad()
    def sample(self, noise, y, cfg_scale: float = 1.0):
        """
        Generation: Gaussian Noise (t=0) -> Synthetic Data (t=1)
        这里用于纯噪声采样 baseline
        """
        x = noise.clone()
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t = 0.0 + i * dt
            x = self.step(x, t, y, dt, cfg_scale=cfg_scale)
        return x

# -----------------------------------------------------------------------------
# 载入 FM 模型
# -----------------------------------------------------------------------------

def load_fm_model(ckpt_path, device, num_classes=7):
    print(f"Loading DiT Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)  # Fix FutureWarning
    
    model = DiT(input_size=32, in_channels=4, num_classes=num_classes).to(device)
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    
    # 优先寻找 EMA 权重
    has_ema = any(k.startswith("ema_module.") for k in state_dict.keys())
    
    if has_ema:
        print("Found EMA weights, loading EMA model for better quality...")
        prefix = "ema_module.module." if "ema_module.module." in list(state_dict.keys())[0] else "ema_module."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_state_dict[new_key] = v
    else:
        print("No EMA weights found, loading standard model weights...")
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[6:]  # remove "model."
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"Warning: Missing keys: {missing}")
    if len(unexpected) > 0:
        unexpected = [k for k in unexpected if "loss" not in k]
        if unexpected:
            print(f"Warning: Unexpected keys (ignored): {unexpected}")
            
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 主流程：按类从噪声直接生成
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained FM .ckpt")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/data0/cyb/ISIC/isic2018_fewshot_latents",
                        help="Directory with .pt latents (few-shot + seed amplification)")
    parser.add_argument("--train_csv", type=str,
                        default="/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv",
                        help="CSV for few-shot amplified set (only用于统计每类已有真实+几何增强样本数)")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/data0/cyb/ISIC/isic_fewshot_augmented_noise",
                        help="Save synthetic images here")
    parser.add_argument("--vae_id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--target_count", type=int, default=536,
                        help="Per-class target count (e.g., max class count in few-shot)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=1.3,
                        help="Classifier-Free Guidance scale (1.0 = no CFG)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.device_id)
    print(f"Running on {device}")

    # 1. Load FM model
    dit = load_fm_model(args.ckpt, device)
    solver = FlowODESolver(dit, num_steps=args.steps, device=device)
    
    # 2. Load VAE
    print(f"Loading VAE: {args.vae_id} ...")
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    vae.eval()

    # 3. 统计 few-shot (含 seed amplification) 各类已有样本数
    print("Counting real (amplified) samples per class...")
    dataset = ISICLatentDataset(args.data_dir, csv_path=args.train_csv)
    
    class_buckets = {}   # 这里可以只存计数，也可以顺便存 latent，不在本脚本中使用
    for i in tqdm(range(len(dataset)), desc="Indexing"):
        item = dataset[i]
        c = int(item["class_labels"])
        l = item["latents"]  # [4, 32, 32]
        
        if c not in class_buckets:
            class_buckets[c] = []
        class_buckets[c].append(l)

    # 转成 tensor 只是为了后续可能 debug 用，本脚本实际只用 len()
    for c in class_buckets:
        class_buckets[c] = torch.stack(class_buckets[c])  # 保持在 CPU 即可
        print(f"Class {c}: {len(class_buckets[c])} real+seed samples")

    # 4. Balanced Generation Loop: 直接从噪声采样
    print(f"Starting Noise-based Generation (Target: {args.target_count} per class)...")
    
    all_classes = sorted(list(class_buckets.keys()))
    if torch.cuda.device_count() > 1:
        my_classes = [c for c in all_classes if c % torch.cuda.device_count() == args.device_id]
    else:
        my_classes = all_classes
    
    print(f"GPU {args.device_id} processing classes: {my_classes}")

    for c_idx in my_classes:
        current_count = len(class_buckets[c_idx])
        n_needed = args.target_count - current_count
        
        if n_needed <= 0:
            print(f"Class {c_idx} already has {current_count} >= {args.target_count}, skipping.")
            continue

        print(f"Class {c_idx}: Generating {n_needed} images from pure noise...")
        save_dir_c = os.path.join(args.output_dir, str(c_idx))
        os.makedirs(save_dir_c, exist_ok=True)

        num_generated = 0
        pbar = tqdm(total=n_needed, desc=f"Class {c_idx}")
        
        while num_generated < n_needed:
            curr_bs = min(args.batch_size, n_needed - num_generated)

            # --- Step A: 直接从 N(0,1) 采噪声 ---
            # 这里的 noise 维度和 FM 训练时的 latent 维度一致：[B, 4, 32, 32]
            noise = torch.randn(curr_bs, 4, 32, 32, device=device)

            y_batch = torch.full((curr_bs,), c_idx, device=device, dtype=torch.long)
            
            # --- Step B: 通过 Flow Matching ODE 从 t=0 -> t=1 生成 latent ---
            z_aug = solver.sample(noise, y_batch, cfg_scale=args.cfg_scale)
            
            # --- Step C: VAE Decode & Save ---
            with torch.no_grad():
                z_aug_recon = z_aug / 0.18215
                imgs = vae.decode(z_aug_recon).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = (imgs.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
            
            for k in range(curr_bs):
                img_pil = Image.fromarray(imgs[k])
                save_path = os.path.join(save_dir_c, f"syn_noise_{c_idx}_{num_generated + k}.jpg")
                img_pil.save(save_path)
            
            num_generated += curr_bs
            pbar.update(curr_bs)
        
        pbar.close()

    print(f"GPU {args.device_id}: Noise-based Generation Finished!")

if __name__ == "__main__":
    main()