import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ISICLatentDataset(Dataset):
    def __init__(self, data_dir, csv_path=None):
        """
        data_dir: 存放所有 .pt 文件的目录
        csv_path: (Optional) 指定要读取的 split csv 文件。
                  如果不传，就读取目录下所有文件（不推荐用于训练，只用于推理或预处理检查）。
        """
        super().__init__()
        self.data_dir = data_dir
        
        if csv_path:
            # 读取 CSV
            df = pd.read_csv(csv_path)
            # 假设 CSV 第一列是 image_id (例如 ISIC_0024306)
            # 我们需要构造对应的文件名列表
            self.files = [f"{img_id}.pt" for img_id in df['image'].values]
            print(f"Loaded {len(self.files)} samples from {csv_path}")
        else:
            # Fallback: 读取目录下所有文件
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
            print(f"Loaded all {len(self.files)} samples from directory (No CSV filter).")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.data_dir, filename)
        
        try:
            # 加载 .pt
            data = torch.load(path, map_location="cpu") # 放在CPU，让DataLoader去搬运
            return {
                "latents": data["latent"].squeeze(0), # [4, 32, 32]
                "class_labels": data["class_idx"]     # int
            }
        except FileNotFoundError:
            print(f"Warning: File {path} not found in precomputed latents!")
            # 简单的容错处理，随机返回一个（实际最好检查数据完整性）
            return self.__getitem__(0)
        


# -----------------------------------------------------------------------------
# Classifier train/inference(严格区分 Train/Val)
# -----------------------------------------------------------------------------

class RealISICImageDataset(Dataset):
    """
    读取真实数据，严格依赖 CSV 文件进行划分。
    """
    def __init__(self, csv_path, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        
        # 读取 CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # 定义类别映射 (ISIC 2018 Task 3 标准)
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        print(f"[RealData] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image']
        
        # 兼容带后缀和不带后缀的情况
        img_name = img_id if str(img_id).endswith('.jpg') else f"{img_id}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            print(f"Warning: Image not found or corrupted: {img_path}")
            # 简单的容错：返回第一张图，避免训练中断 (实际部署可更严格)
            return self.__getitem__(0)

        if self.transform:
            image = self.transform(image)
            
        # 解析 One-hot Label
        # 假设 CSV 格式: image, MEL, NV, ... (数值为 0.0 或 1.0)
        labels = row[self.class_names].values.astype(float)
        label_idx = int(labels.argmax())
        
        return image, label_idx

class SyntheticISICImageDataset(Dataset):
    """
    读取生成的增强数据。
    目录结构假设: root_dir/0/img.jpg, root_dir/1/img.jpg ...
    或者 root_dir/class_0/img.jpg
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if os.path.exists(root_dir):
            # 遍历所有子文件夹
            for class_folder in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_folder)
                if os.path.isdir(class_path):
                    # 尝试解析类别 ID
                    try:
                        # 兼容 "0" 或 "class_0" 这种命名
                        class_id = int(class_folder.split('_')[-1]) if '_' in class_folder else int(class_folder)
                    except ValueError:
                        continue # 跳过无法解析的文件夹

                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append((os.path.join(class_path, img_name), class_id))
        else:
            print(f"[SyntheticData] Warning: Directory {root_dir} does not exist.")

        print(f"[SyntheticData] Loaded {len(self.samples)} augmented samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading synthetic image {path}: {e}")
            return self.__getitem__(0)

        if self.transform:
            image = self.transform(image)
        return image, label