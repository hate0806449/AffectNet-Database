import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AffectNetDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        
        self.image_files = sorted(os.listdir(images_dir))  # 0.jpg, 1.jpg, 2.jpg ...

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]  # '0.jpg'
        prefix = os.path.splitext(img_name)[0]  # 取 '0'
        
        # 讀圖片
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 讀標註
        exp_path = os.path.join(self.annotations_dir, f"{prefix}_exp.npy")
        exp_label = np.load(exp_path).item()  # 假設裡面是純標籤數字，用 item() 拿出

        # 如果需要其他標註檔也可同理讀取
        # lnd = np.load(os.path.join(self.annotations_dir, f"{prefix}_lnd.npy"))
        # val = np.load(os.path.join(self.annotations_dir, f"{prefix}_val.npy"))
        # aro = np.load(os.path.join(self.annotations_dir, f"{prefix}_aro.npy"))

        if self.transform:
            image = self.transform(image)

        return image, int(exp_label)
