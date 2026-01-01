import argparse
import torch
import os
import sys
import math
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from tqdm import tqdm

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import DiT

def load_model_from_checkpoint(ckpt_path, device, image_size=32, num_classes=7):
    """
    从 Lightning Checkpoint 加载 DiT 权重
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 初始化模型结构
    model = DiT(
        input_size=image_size,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes
    ).to(device)
    
    # 处理 Lightning 的 state_dict 键名 (移除 "model." 前缀)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        # 如果你用了 EMA，且只想加载 EMA 权重，需要在这里处理
        # 通常 Lightning Checkpoint 存的是主模型权重
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

@torch.no_grad()
def ode_sample(model, z, y, num_steps=50, cfg_scale=4.0, device="cuda"):
    """
    使用 Euler 方法求解 ODE: dx/dt = v(x, t)
    t 从 0 (Noise) -> 1 (Data)
    """
    B = z.shape[0]
    dt = 1.0 / num_steps
    
    # 构造无条件生成的 Label (用于 CFG)
    # 假设 DiT 内部 num_classes 是 CFG 的 null token
    # 我们在 DiT 定义时设置了 num_classes=7, 所以 7 就是 null token
    y_null = torch.full_like(y, fill_value=7, device=device)
    
    # Euler Integration
    images = z.clone()
    for i in tqdm(range(num_steps), desc="Sampling"):
        t = i / num_steps
        t_tensor = torch.ones(B, device=device) * t
        
        # 1. Conditional Prediction
        v_cond = model(images, t_tensor, y)
        
        # 2. Unconditional Prediction (for CFG)
        if cfg_scale > 1.0:
            v_uncond = model(images, t_tensor, y_null)
            # CFG 公式: v = v_uncond + scale * (v_cond - v_uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
            
        # 3. Step (x_next = x + v * dt)
        images = images + v * dt
        
    return images

def main():
    parser = argparse.ArgumentParser()
    # 你的 checkpoint 路径
    parser.add_argument("--ckpt", type=str, required=True, help="Path to lightning .ckpt file", default="/home/chenyibiao/MFM-II/checkpoints/isic_fm_dit_v2/epoch=384-val/loss=1.7222.ckpt")
    parser.add_argument("--output_dir", type=str, default="results/inference_fm/ISIC")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load VAE (用于解码)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
    vae.eval()
    
    # 2. Load DiT
    model = load_model_from_checkpoint(args.ckpt, args.device)
    
    # 3. 准备采样条件
    # 7个类别，每个类别生成 4 张图
    num_classes = 7
    images_per_class = 4
    total_images = num_classes * images_per_class
    
    # 构造 Labels: [0,0,0,0, 1,1,1,1, ..., 6,6,6,6]
    labels = torch.arange(num_classes, device=args.device).repeat_interleave(images_per_class)
    
    # 构造初始噪声 x_0 ~ N(0, 1) [B, 4, 32, 32]
    z_0 = torch.randn(total_images, 4, 32, 32, device=args.device)
    
    # 4. 执行 Flow Matching 采样
    print(f"Generating {total_images} images with CFG={args.cfg_scale}...")
    latents = ode_sample(model, z_0, labels, num_steps=args.steps, cfg_scale=args.cfg_scale, device=args.device)
    
    # 5. VAE Decode
    print("Decoding latents...")
    with torch.no_grad():
        # 记得除以 scaling factor !
        latents = latents / 0.18215
        imgs = vae.decode(latents).sample
    
    # 6. Post-process & Save
    # VAE 输出是 [-1, 1]，需要转回 [0, 1]
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    
    save_path = os.path.join(args.output_dir, "sample_grid.png")
    save_image(imgs, save_path, nrow=images_per_class, padding=2)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()