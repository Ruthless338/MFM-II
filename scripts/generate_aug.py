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

class FlowODESolver:
    """
    Flow Matching 的核心：ODE 求解器
    支持正向生成 (0 -> 1) 和反向 Inversion (1 -> 0)
    """
    def __init__(self, model, num_steps=50, device="cuda"):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        
    @torch.no_grad()
    def step(self, x, t, y, dt):
        """
        Euler 方法单步积分: x_{t+dt} = x_t + v(x_t, t) * dt
        """
        # 广播时间 t
        t_tensor = torch.ones(x.shape[0], device=self.device) * t
        
        # 预测速度场 v
        v_pred = self.model(x, t_tensor, y)
        
        # 更新 x
        x_new = x + v_pred * dt
        return x_new

    @torch.no_grad()
    def invert(self, latents, y):
        """
        Inversion: Real Data (t=1) -> Gaussian Noise (t=0)
        这是 Diff-II 的 DDIM Inversion 的 Flow Matching 版本。
        因为 FM 轨迹是直的，所以这里的 Inversion 误差比 Diffusion 小得多。
        """
        x = latents.clone()
        dt = -1.0 / self.num_steps # 时间倒流
        
        # 从 t=1 积分到 t=0
        for i in range(self.num_steps):
            t = 1.0 + i * dt 
            x = self.step(x, t, y, dt)
            
        return x 

    @torch.no_grad()
    def sample(self, noise, y):
        """
        Generation: Gaussian Noise (t=0) -> Synthetic Data (t=1)
        """
        x = noise.clone()
        dt = 1.0 / self.num_steps # 时间正向
        
        # 从 t=0 积分到 t=1
        for i in range(self.num_steps):
            t = 0.0 + i * dt
            x = self.step(x, t, y, dt)
            
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
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained FM .ckpt file")
    parser.add_argument("--data_dir", type=str, default="/mnt/data0/cyb/ISIC/isic2018_latents_256", help="Precomputed latents dir")
    parser.add_argument("--train_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/train_split.csv")
    parser.add_argument("--output_dir", type=str, default="/mnt/data0/cyb/ISIC/isic_augmented_256", help="Where to save images")
    parser.add_argument("--vae_id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50, help="ODE steps for inversion/sampling")
    parser.add_argument("--device_id", type=int, default=0, help="GPU Device ID (0-3)")
    parser.add_argument("--multiplier", type=int, default=5, help="每张原图生成多少张增强图")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ''' 以下GPU分配策略仅适用于ISIC的七类分配 '''
    # 设置当前设备
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.device_id)
    print(f"Running on {device}")
    
    # 定义类别分配策略
    # 类别 1 是最大类，分配给 GPU 0
    # 其他类别两两分组：(0, 2) -> GPU 1, (3, 4) -> GPU 2, (5, 6) -> GPU 3
    # 注意：这里假设类别 ID 是 0-6。你需要确认 CSV 中的 label 映射关系。
    # 根据之前的脚本，class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    # 0: MEL, 1: NV, 2: BCC, 3: AKIEC, 4: BKL, 5: DF, 6: VASC
    
    target_classes = []
    if args.device_id == 0:
        target_classes = [1] # NV (最大类)
    elif args.device_id == 1:
        target_classes = [0, 2] # MEL, BCC
    elif args.device_id == 2:
        target_classes = [3, 4] # AKIEC, BKL
    elif args.device_id == 3:
        target_classes = [5, 6] # DF, VASC
    else:
        print(f"Device ID {args.device_id} out of range (0-3). Exiting.")
        return

    print(f"GPU {args.device_id} is responsible for classes: {target_classes}")

    # 1. 加载模型
    dit = load_fm_model(args.ckpt, device)
    solver = FlowODESolver(dit, num_steps=args.steps, device=device)
    
    print(f"Loading VAE: {args.vae_id} ...")
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    vae.eval()

    # 2. 读取所有真实 Latents 并按类别分组
    # 这一步是为了方便做 Intra-class Pairing
    print("Grouping real data by class...")
    dataset = ISICLatentDataset(args.data_dir, csv_path=args.train_csv)
    
    # 字典结构: { class_idx: Tensor[N, 4, 32, 32] }
    class_buckets = {} 
    
    for i in tqdm(range(len(dataset)), desc="Loading latents"):
        item = dataset[i]
        c = item["class_labels"]
        l = item["latents"]
        
        # 只加载当前 GPU 负责的类别，节省内存
        if c in target_classes:
            if c not in class_buckets:
                class_buckets[c] = []
            class_buckets[c].append(l)

    # 将 List 转为 Tensor
    for c in class_buckets:
        if len(class_buckets[c]) > 0:
            class_buckets[c] = torch.stack(class_buckets[c]).to(device)
            print(f"Class {c}: {len(class_buckets[c])} samples loaded on GPU {args.device_id}")
        else:
            print(f"Warning: Class {c} has no samples!")

    # 3. 开始增强循环 (Augmentation Loop)
    print(f"Starting Augmentation (Multiplier x{args.multiplier})...")
    
    for c_idx in target_classes:
        if c_idx not in class_buckets:
            continue
            
        latents = class_buckets[c_idx]
        print(f"Processing Class {c_idx} on GPU {args.device_id}...")
        save_dir_c = os.path.join(args.output_dir, str(c_idx)) # 按类别分文件夹存
        os.makedirs(save_dir_c, exist_ok=True)
        
        num_real = len(latents)
        total_to_gen = num_real * args.multiplier
        
        num_done = 0
        pbar = tqdm(total=total_to_gen, desc=f"Class {c_idx}")
        
        while num_done < total_to_gen:
            curr_bs = min(args.batch_size, total_to_gen - num_done)
            
            # --- Step A: Random Pairing (随机配对) ---
            idx1 = torch.randint(0, num_real, (curr_bs,), device=device)
            idx2 = torch.randint(0, num_real, (curr_bs,), device=device)
            
            z1_real = latents[idx1]
            z2_real = latents[idx2]
            
            y_batch = torch.full((curr_bs,), c_idx, device=device, dtype=torch.long)
            
            # --- Step B: Inversion (Real -> Noise) ---
            z1_noise = solver.invert(z1_real, y_batch)
            z2_noise = solver.invert(z2_real, y_batch)
            
            # --- Step C: Inversion Circle Interpolation (Diff-II Core) ---
            z_mix_noise = rand_slerp(z1_noise, z2_noise, alpha=None)
            
            # --- Step D: Generation (Noise -> Synthetic Data) ---
            z_aug = solver.sample(z_mix_noise, y_batch)
            
            # --- Step E: VAE Decode & Save ---
            with torch.no_grad():
                z_aug_recon = z_aug / 0.18215
                imgs = vae.decode(z_aug_recon).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = (imgs.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
                
            for k in range(curr_bs):
                img_pil = Image.fromarray(imgs[k])
                save_path = os.path.join(save_dir_c, f"aug_{num_done + k}.jpg")
                img_pil.save(save_path)
            
            num_done += curr_bs
            pbar.update(curr_bs)
            
        pbar.close()

    print(f"GPU {args.device_id}: All augmentations finished!")

if __name__ == "__main__":
    main()