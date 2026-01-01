from torchmetrics import Accuracy, F1Score
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# -----------------------------------------------------------------------------
# Classifier Lightning Module
# -----------------------------------------------------------------------------

class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=7, lr=1e-3, weight_decay=1e-4, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        
        # 使用 torchvision 官方预训练权重
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 替换全连接层适应 ISIC 类别数
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics: 关注 Macro F1 (因为长尾)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log: on_step=True 方便看 loss 曲线细节
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_macro', self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 经典的 SGD + Momentum + Cosine Decay 组合，适合 ResNet
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.hparams.lr, 
            momentum=0.9, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]