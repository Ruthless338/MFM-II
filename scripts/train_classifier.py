import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import ResNetClassifier
from data.dataset import RealISICImageDataset, SyntheticISICImageDataset

def main():
    parser = argparse.ArgumentParser()
    # Data Paths
    parser.add_argument("--real_img_dir", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input")
    parser.add_argument("--train_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/train_split.csv", help="Split data generated script")
    parser.add_argument("--val_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv")
    parser.add_argument("--val_img_dir", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input")
    parser.add_argument("--syn_data_dir", type=str, default="/mnt/data0/cyb/ISIC/isic_augmented_256")
    
    # Experiment Settings
    parser.add_argument("--exp_name", type=str, default="baseline_resnet50", help="WandB run name")
    parser.add_argument("--use_synthetic", action="store_true", help="Merge synthetic data into training")
    parser.add_argument("--seed", type=int, default=42)
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.01) # SGD 常用 0.01 或 0.001
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=4, help="Number of GPUs")
    
    args = parser.parse_args()

    # 1. Setup Seed
    pl.seed_everything(args.seed, workers=True)

    # 2. Transforms (ImageNet Stats)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # 医学皮肤图垂直翻转也是合理的增强
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Prepare Datasets
    # 3.1 Real Train
    real_train_ds = RealISICImageDataset(args.train_csv, args.real_img_dir, transform=train_transform)
    
    # 3.2 Real Validation (纯净验证集，决不能包含合成数据)
    real_val_ds = RealISICImageDataset(args.val_csv, args.val_img_dir, transform=val_transform)
    
    # 3.3 Combine with Synthetic (Optional)
    if args.use_synthetic:
        syn_ds = SyntheticISICImageDataset(args.syn_data_dir, transform=train_transform)
        if len(syn_ds) > 0:
            final_train_ds = ConcatDataset([real_train_ds, syn_ds])
            print(f"--> Experiment: Training on Real ({len(real_train_ds)}) + Synthetic ({len(syn_ds)})")
        else:
            print("--> Warning: No synthetic data found, falling back to Real only.")
            final_train_ds = real_train_ds
    else:
        final_train_ds = real_train_ds
        print(f"--> Experiment: Baseline (Real data only, {len(real_train_ds)} samples)")

    # 4. DataLoaders
    # DDP 模式下，batch_size 是每张卡的 size
    train_loader = DataLoader(
        final_train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True # 提升训练效率
    )
    
    val_loader = DataLoader(
        real_val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 5. WandB Logger
    wandb_logger = WandbLogger(
        project="ISIC-FlowMatching-Aug",
        name=args.exp_name,
        log_model=True,
        config=vars(args)
    )

    # 6. Model
    model = ResNetClassifier(
        num_classes=7, 
        lr=args.lr, 
        max_epochs=args.epochs
    )

    # 7. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/classifier/{args.exp_name}",
        filename="best-{epoch:02d}-{val/f1_macro:.4f}",
        monitor="val/f1_macro", # 监控 Macro F1
        mode="max",
        save_top_k=1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 8. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        precision="16-mixed", # 混合精度，显存更省，速度更快
        log_every_n_steps=10,
        sync_batchnorm=True, # 如果显存够，建议开启，对 BN 稳定性有帮助
    )

    # 9. Fit
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()