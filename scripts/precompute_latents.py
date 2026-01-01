import argparse
import os
import sys
import yaml
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL

# 添加项目根目录到 path，防止 import 报错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ISICRawDataset(Dataset):
    def __init__(self, image_dir, csv_path, resolution=256, ext=".jpg"):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.ext = ext
        
        # 定义类别列名
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        # 图像预处理：Resize -> CenterCrop (保证正方形) -> ToTensor -> Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Map [0, 1] to [-1, 1] for VAE
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image']
        
        # 1. Load Image
        image_path = os.path.join(self.image_dir, f"{image_id}{self.ext}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        
        # 2. Extract Label (One-hot -> Index)
        # 找到哪一列是 1.0
        # row[self.class_names] 会返回一个 Series，values 是 numpy array
        labels = row[self.class_names].values.astype(float)
        class_idx = np.argmax(labels) # 比如 [0, 1, 0...] -> 1
        
        return {
            "pixel_values": pixel_values,
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "image_id": image_id
        }

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/preprocess_isic.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # 1. Setup Device & Create Output Dir
    device = torch.device(cfg["vae"]["device"])
    os.makedirs(cfg["dataset"]["output_dir"], exist_ok=True)
    
    # 2. Load VAE
    print(f"Loading VAE model: {cfg['vae']['model_id']}...")
    vae = AutoencoderKL.from_pretrained(cfg["vae"]["model_id"]).to(device)
    vae.eval()
    vae.requires_grad_(False) # 冻结 VAE
    
    # 3. Prepare DataLoader
    print("Preparing dataset...")
    dataset = ISICRawDataset(
        image_dir=cfg["dataset"]["image_dir"],
        csv_path=cfg["dataset"]["csv_path"],
        resolution=cfg["dataset"]["resolution"],
        ext=cfg["dataset"]["ext"]
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg["vae"]["batch_size"], 
        shuffle=False, 
        num_workers=cfg["vae"]["num_workers"],
        pin_memory=True
    )
    
    print(f"Start processing {len(dataset)} images...")
    
    # 4. Processing Loop
    scaling_factor = cfg["vae"]["scaling_factor"]
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["pixel_values"].to(device)
            class_idxs = batch["class_idx"] # CPU tensor is fine for saving
            image_ids = batch["image_id"]
            
            # Encode to Latent Distribution
            # posterior 是一个 DiagonalGaussianDistribution 对象
            posterior = vae.encode(images).latent_dist
            
            # Sample from the distribution (reparameterization trick)
            # 或者使用 .mode() 取均值，但 sample() 对于生成模型通常更好，增加鲁棒性
            latents = posterior.sample() 
            
            # Scale Latents (Critical for SD VAE!)
            latents = latents * scaling_factor
            
            # Save Batch
            latents = latents.cpu()
            
            for i in range(len(image_ids)):
                save_path = os.path.join(cfg["dataset"]["output_dir"], f"{image_ids[i]}.pt")
                torch.save({
                    "latent": latents[i].clone(),      # Shape: [4, 32, 32]
                    "class_idx": class_idxs[i].item(), # Int
                    "image_id": image_ids[i]           # Str
                }, save_path)
                
    print(f"Done! Processed data saved to {cfg['dataset']['output_dir']}")
    print(f"Latent shape example: {latents[0].shape}") # Should be [4, 32, 32] for 256x256 image

if __name__ == "__main__":
    main()