import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from copy import deepcopy
from models.dit import DiT
import torch.nn as nn

# 简单的 EMA 帮助类
class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.decay == 1: # 刚开始可能不需要 decay
                    ema_v.copy_(model_v)
                else:
                    update_fn(ema_v, model_v)

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class FMLightningModule(pl.LightningModule):
    def __init__(self, 
                 lr=1e-4, 
                 num_classes=7, 
                 image_size=32,
                 weight_decay=0.01,
                 max_epochs=500,
                 use_ema=True):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DiT(
            input_size=image_size,
            patch_size=2,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes
        )
        
        # 初始化 EMA 模型
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = EMA(self.model, decay=0.9999)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 每个 step 更新一次 EMA
        if self.use_ema:
            self.ema_model.update(self.model)

    def training_step(self, batch, batch_idx):
        x_1 = batch["latents"]
        y = batch["class_labels"]
        
        # Flow Matching Logic
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=self.device)
        t_expand = t.view(-1, 1, 1, 1)
        
        # Optimal Transport Path (Linear)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0 # Target Velocity
        
        v_pred = self.model(x_t, t, y)
        loss = F.mse_loss(v_pred, u_t)
        
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证集也算一样的 Loss，用于监控过拟合
        # 注意：这里我们应该用 EMA 模型来验证吗？通常是的，但为了简单，这里先用主模型
        # 或者你可以切换到 self.ema_model.module 来做 forward
        x_1 = batch["latents"]
        y = batch["class_labels"]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=self.device)
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0
        
        # 使用 EMA 模型进行验证（如果有）
        if self.use_ema:
            v_pred = self.ema_model.module(x_t, t, y)
        else:
            v_pred = self.model(x_t, t, y)
            
        loss = F.mse_loss(v_pred, u_t)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine Scheduler with Warmup
        # 假设 dataloader 长度未知，这里用 total_steps 估算或者直接依赖 max_epochs
        # Lightning 会自动处理 steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        
        return [optimizer], [scheduler]