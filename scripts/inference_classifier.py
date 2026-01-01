import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import shutil

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import ResNetClassifier

# -----------------------------------------------------------------------------
# 1. Dataset & Utils
# -----------------------------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        labels = row[self.class_names].values.astype(float)
        label_idx = int(labels.argmax())
        
        return image, label_idx, img_name  # 返回 img_name 用于错误分析

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

# -----------------------------------------------------------------------------
# 2. Visualization Functions
# -----------------------------------------------------------------------------

def plot_confusion_matrix_scientific(cm, class_names, save_path):
    """
    绘制更科学、更易读的混淆矩阵
    """
    # Normalize by row (True label) -> Recall
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    # 使用 Blues 配色，适合医学图像
    # annot_kws 设置字体大小
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10}, cbar_kws={'label': 'Recall'})
    
    plt.ylabel('True Class', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.title('Normalized Confusion Matrix', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix saved to {save_path}")
    plt.close()

def plot_radar_chart(report_dict, class_names, save_path):
    """
    绘制雷达图，展示每个类别的 F1-Score
    这对于 Long-tail 问题非常直观，能看到尾部类别的提升
    """
    # 提取 F1-Score
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
    
    # 闭合数据
    values = f1_scores + [f1_scores[0]]
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += [angles[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], class_names)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='F1-Score')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title('Per-Class F1 Score', fontweight='bold', y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(save_path, dpi=300)
    print(f"Radar Chart saved to {save_path}")
    plt.close()

def plot_tsne(features, labels, class_names, save_path, limit=2000):
    """
    绘制 t-SNE 特征分布图
    """
    print("Computing t-SNE...")
    # 如果数据太多，随机采样 limit 个
    if len(labels) > limit:
        idx = np.random.choice(len(labels), limit, replace=False)
        features = features[idx]
        labels = labels[idx]
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    
    # 使用 tab10 调色板
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        idxs = labels == i
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], 
                    color=colors[i], label=class_name, alpha=0.6, s=20)
        
    plt.legend(loc='best', fontsize='medium')
    plt.title('t-SNE Visualization of Feature Space', fontweight='bold')
    plt.axis('off') # 隐藏坐标轴
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE Plot saved to {save_path}")
    plt.close()

# -----------------------------------------------------------------------------
# 3. Main Evaluation Logic
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, class_names, output_dir, img_source_dir):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = [] 
    all_img_names = []
    all_features = [] # 存储倒数第二层特征
    
    # Hook to get features (ResNet50: avgpool layer)
    features_container = []
    def hook_fn(module, input, output):
        features_container.append(output.flatten(1).cpu().numpy())
    
    # Register hook (resnet50.avgpool is the layer before fc)
    # LightningModule 包装了一层，所以是 model.model.avgpool
    handle = model.model.avgpool.register_forward_hook(hook_fn)
    
    print("Running Inference...")
    for imgs, labels, img_names in tqdm(dataloader):
        imgs = imgs.to(device)
        
        # Forward
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_img_names.extend(img_names)
        
        # Hook 会自动填充 features_container
    
    # Remove hook
    handle.remove()
    
    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_features = np.concatenate(features_container, axis=0)
    
    # --- 1. Metrics Calculation ---
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Save Metrics to txt
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
        
    print(f"Metrics saved. F1 Macro: {f1_macro:.4f}")
    
    # --- 2. Classification Report ---
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    # --- 3. Plots ---
    set_style()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix_scientific(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    
    # Radar Chart
    plot_radar_chart(report, class_names, os.path.join(output_dir, "radar_chart.png"))
    
    # t-SNE
    plot_tsne(all_features, all_labels, class_names, os.path.join(output_dir, "tsne_plot.png"))
    
    # --- 4. Error Analysis (Top-k Hardest Failures) ---
    # 找出模型最自信但预测错误的样本
    print("Generating Error Analysis...")
    error_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)
    
    # 计算每个样本的 confidence (predicted probability)
    confidences = np.max(all_probs, axis=1)
    
    # 筛选出错误的样本
    incorrect_mask = all_preds != all_labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # 按置信度降序排列 (最自信的错误)
    sorted_incorrect_indices = incorrect_indices[np.argsort(-confidences[incorrect_indices])]
    
    # 保存前 20 个错误样本
    top_k = 20
    for i, idx in enumerate(sorted_incorrect_indices[:top_k]):
        img_name = all_img_names[idx]
        true_cls = class_names[all_labels[idx]]
        pred_cls = class_names[all_preds[idx]]
        conf = confidences[idx]
        
        src_file = os.path.join(img_source_dir, f"{img_name}.jpg")
        dst_file = os.path.join(error_dir, f"{i+1}_{true_cls}_pred_{pred_cls}_{conf:.2f}.jpg")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            
    print(f"Top-{top_k} error images saved to {error_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_img_dir", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input")
    parser.add_argument("--test_csv", type=str, default="/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
    parser.add_argument("--output_dir", type=str, default="results/inference_v2")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading checkpoint: {args.ckpt_path}")
    try:
        model = ResNetClassifier.load_from_checkpoint(args.ckpt_path, num_classes=args.num_classes)
    except Exception:
        model = ResNetClassifier(num_classes=args.num_classes)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt['state_dict'])
    
    model.to(device)
    model.eval()
    
    # Data
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(args.test_img_dir, args.test_csv, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    # Run
    evaluate(model, test_loader, device, class_names, args.output_dir, args.test_img_dir)

if __name__ == "__main__":
    main()