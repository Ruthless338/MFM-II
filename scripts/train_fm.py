import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy 
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import ISICLatentDataset
from models.lightning_fm import FMLightningModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/data0/cyb/ISIC/isic2018_latents_256")
    parser.add_argument("--train_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/train_split.csv") 
    parser.add_argument("--val_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # 1. Prepare Data
    # 训练集：只读 train_split.csv
    train_dataset = ISICLatentDataset(
        data_dir=args.data_dir, 
        csv_path=args.train_csv
    )
    
    # 验证集：只读 val_split.csv
    val_dataset = ISICLatentDataset(
        data_dir=args.data_dir, 
        csv_path=args.val_csv
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True,
        drop_last=True # 训练时丢弃最后一个不完整的batch有助于稳定
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True
    )

    # 2. Model
    model = FMLightningModule(
        lr=args.lr, 
        num_classes=7, 
        image_size=32, 
        max_epochs=args.max_epochs
    )

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/isic_fm_dit_v2", # 改个名字区分版本
        filename="{epoch:03d}-{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss", # 监控验证集 Loss
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,            # 通用写法
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.max_epochs,
        precision="16-mixed", # 必须开，DiT对精度不敏感但对显存敏感
        logger = WandbLogger(project="MFM-II", name="train_fm_0"),
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0 # 增加梯度裁剪，防止 Transformer 训练不稳定
    )

    # 5. Start Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()