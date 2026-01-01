# scripts/generate_aug_fewshot.py
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
'''
import argparse
import torch
import os
import sys
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import DiT
from data.dataset import ISICLatentDataset

# -----------------------------------------------------------------------------
# 1. 数学核心工具 (Math Utils)
# -----------------------------------------------------------------------------

# 球面插值
def rand_slerp(z1, z2, alpha=None):
    """
    球面线性插值 (Spherical Linear Interpolation)
    复用 Diff-II 的核心逻辑，但在 Batch 维度上进行了优化。
    z1, z2: [B, C, H, W]
    alpha: 插值比例，如果为 None 则随机采样
    """
    B, C, H, W = z1.shape
    flat_z1 = z1.view(B, -1)
    flat_z2 = z2.view(B, -1)
    
    # 计算两个向量的夹角 theta
    # normalize 是为了计算角度，插值时用原始模长
    z1_n = flat_z1 / torch.norm(flat_z1, dim=1, keepdim=True)
    z2_n = flat_z2 / torch.norm(flat_z2, dim=1, keepdim=True)
    
    # dot product with numerical stability clamp
    dot = (z1_n * z2_n).sum(dim=1, keepdim=True)
    dot = torch.clamp(dot, -0.9995, 0.9995)
    
    theta = torch.acos(dot)
    
    if alpha is None:
        # Diff-II logic: alpha ~ Uniform(0, 1) or specific distribution
        # 这里我们在 [0.2, 0.8] 之间采样，避免生成和原图一模一样的图
        alpha = torch.rand(B, 1, device=z1.device) * 0.6 + 0.2
    else:
        alpha = torch.ones(B, 1, device=z1.device) * alpha

    # Slerp 公式
    sin_theta = torch.sin(theta)
    scale1 = torch.sin((1 - alpha) * theta) / sin_theta
    scale2 = torch.sin(alpha * theta) / sin_theta
    
    # Reshape scales to [B, 1, 1, 1] for broadcasting
    scale1 = scale1.view(B, 1, 1, 1)
    scale2 = scale2.view(B, 1, 1, 1)
    
    z_mix = scale1 * z1 + scale2 * z2
    return z_mix

# 线性插值 (模拟 Diff-Mix 的逻辑)
def rand_linear(z1, z2, alpha=None):
    B = z1.shape[0]
    if alpha is None:
        alpha = torch.rand(B, 1, 1, 1, device=z1.device) * 0.6 + 0.2
    else:
        alpha = torch.ones(B, 1, 1, 1, device=z1.device) * alpha
    
    # 简单的线性混合
    return (1 - alpha) * z1 + alpha * z2

class FlowODESolver:
    """
    Flow Matching 的核心：ODE 求解器
    - Inversion: t=1 -> 0（建议不使用 CFG）
    - Sampling : t=0 -> 1（可使用 CFG）
    数值积分：Heun（二阶预测-校正），通常比 Euler 更稳、同步数质量更好
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
    def invert(self, latents, y):
        """
        Inversion: Real Data (t=1) -> Gaussian Noise (t=0)
        通常 inversion 不开 CFG（cfg_scale=1）
        """
        x = latents.clone()
        dt = -1.0 / self.num_steps

        for i in range(self.num_steps):
            t = 1.0 + i * dt
            x = self.step(x, t, y, dt, cfg_scale=1.0)
        return x

    @torch.no_grad()
    def sample(self, noise, y, cfg_scale: float = 1.0):
        """
        Generation: Gaussian Noise (t=0) -> Synthetic Data (t=1)
        可开启 CFG
        """
        x = noise.clone()
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t = 0.0 + i * dt
            x = self.step(x, t, y, dt, cfg_scale=cfg_scale)
        return x

# -----------------------------------------------------------------------------
# 2. 主流程
# -----------------------------------------------------------------------------

def load_fm_model(ckpt_path, device, num_classes=7):
    print(f"Loading DiT Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False) # Fix FutureWarning
    
    # 初始化 DiT
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
                # 去掉前缀，例如 "ema_module.x_embedder.proj.weight" -> "x_embedder.proj.weight"
                new_key = k[len(prefix):]
                new_state_dict[new_key] = v
    else:
        print("No EMA weights found, loading standard model weights...")
        # 如果没有 EMA，尝试加载普通权重 (model.xxx)
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[6:] # remove "model."
                new_state_dict[new_key] = v
            else:
                # 兼容性处理：有些 key 可能直接就是 layer 名
                new_state_dict[k] = v

    # 加载权重
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"Warning: Missing keys: {missing}")
    if len(unexpected) > 0:
        # 过滤掉 loss 相关的 key，通常不需要
        unexpected = [k for k in unexpected if "loss" not in k]
        if unexpected:
            print(f"Warning: Unexpected keys (ignored): {unexpected}")
            
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained FM .ckpt")
    parser.add_argument("--data_dir", type=str, default="/mnt/data0/cyb/ISIC/isic2018_fewshot_latents", help="Directory with .pt latents")
    parser.add_argument("--train_csv", type=str, default="/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv")
    parser.add_argument("--output_dir", type=str, default="/mnt/data0/cyb/ISIC/isic_fewshot_augmented", help="Save synthetic images here")
    parser.add_argument("--vae_id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--target_count", type=int, default=536)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--interpolation_mode", type=str, default="Slerp", choices=["Slerp", "Linear"], help="Interpolation Comparison modes")
    parser.add_argument("--cfg_scale", type=float, default=1.8, help="Classifier-Free Guidance scale (1.0 = no CFG)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.device_id)
    print(f"Running on {device}")

    # 1. Load Models
    dit = load_fm_model(args.ckpt, device)
    solver = FlowODESolver(dit, num_steps=args.steps, device=device)
    
    print(f"Loading VAE: {args.vae_id} ...")
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    vae.eval()

    # 2. Group Real Latents by Class
    # 这里的 Dataset 会加载 amplifed.csv 里的所有数据（包含 Flip/Rotate 后的样本）
    print("Grouping real (amplified) latents by class...")
    dataset = ISICLatentDataset(args.data_dir, csv_path=args.train_csv)
    
    class_buckets = {} 
    for i in tqdm(range(len(dataset)), desc="Indexing"):
        item = dataset[i]
        c = item["class_labels"]
        l = item["latents"] # [4, 32, 32]
        
        if c not in class_buckets:
            class_buckets[c] = []
        class_buckets[c].append(l)

    # Convert lists to tensors
    for c in class_buckets:
        class_buckets[c] = torch.stack(class_buckets[c]).to(device)
        print(f"Class {c}: {len(class_buckets[c])} seed samples")

    # 3. Balanced Generation Loop
    print(f"Starting Balanced Generation (Target: {args.target_count})...")
    
    # 获取所有类别 ID
    all_classes = sorted(list(class_buckets.keys()))
    
    # 动态 GPU 分配：简单的取模分配
    my_classes = [c for c in all_classes if c % torch.cuda.device_count() == args.device_id] if torch.cuda.device_count() > 1 else all_classes
    
    print(f"GPU {args.device_id} processing classes: {my_classes}")

    for c_idx in my_classes:
        latents = class_buckets[c_idx]
        current_count = len(latents)
        
        # 计算需要生成的数量：目标数量 - 现有数量
        # 注意：这里的现有数量已经包含了 Seed Amplification 的 5倍扩充
        # 例如 DF: 原本9张 -> 扩充后45张。需要生成 536 - 45 = 491 张。
        n_needed = args.target_count - current_count
        
        if n_needed <= 0:
            print(f"Class {c_idx} already has {current_count} >= {args.target_count}, skipping.")
            continue
            
        print(f"Class {c_idx}: Generating {n_needed} images...")
        
        save_dir_c = os.path.join(args.output_dir, str(c_idx))
        os.makedirs(save_dir_c, exist_ok=True)
        
        num_generated = 0
        pbar = tqdm(total=n_needed, desc=f"Class {c_idx}")
        
        while num_generated < n_needed:
            curr_bs = min(args.batch_size, n_needed - num_generated)
            
            # --- Step A: Random Pairing from Amplified Pool ---
            # 因为池子变大了，随机配对的组合指数级增加，这就是 Seed Amplification 的意义
            idx1 = torch.randint(0, len(latents), (curr_bs,), device=device)
            idx2 = torch.randint(0, len(latents), (curr_bs,), device=device)
            
            z1_real = latents[idx1]
            z2_real = latents[idx2]
            
            y_batch = torch.full((curr_bs,), c_idx, device=device, dtype=torch.long)
            
            # --- Step B: Inversion (Real -> Noise) ---
            z1_noise = solver.invert(z1_real, y_batch)
            z2_noise = solver.invert(z2_real, y_batch)
            
            # --- Step C: Inversion Interpolation ---
            if args.interpolation_mode == "Linear":
                # 使用线性插值 (Linear)
                z_mix_noise = rand_linear(z1_noise, z2_noise)
            else:
                # 使用球面插值 (Slerp)
                z_mix_noise = rand_slerp(z1_noise, z2_noise)
            
            # --- Step D: Generation (Noise -> Synthetic Data) ---
            z_aug = solver.sample(z_mix_noise, y_batch, cfg_scale=args.cfg_scale)
            
            # --- Step E: VAE Decode & Save ---
            with torch.no_grad():
                z_aug_recon = z_aug / 0.18215
                imgs = vae.decode(z_aug_recon).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = (imgs.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
                
            for k in range(curr_bs):
                img_pil = Image.fromarray(imgs[k])
                # 命名加上 class 和 index 防止冲突
                save_path = os.path.join(save_dir_c, f"syn_{c_idx}_{num_generated + k}.jpg")
                img_pil.save(save_path)
            
            num_generated += curr_bs
            pbar.update(curr_bs)
            
        pbar.close()

    print(f"GPU {args.device_id}: Generation Finished!")

if __name__ == "__main__":
    main()