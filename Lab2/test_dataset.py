import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
import random

class RoadSegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.tiff")))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_filename = img_path.name.replace(".tiff", ".tif")
        label_path = self.labels_dir / label_filename
        
        image = tiff.imread(img_path)
        label = tiff.imread(label_path)
        
        # Нормализация изображения
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Приведение label к бинарному виду
        if label.max() > 1:
            label = (label > 0).astype(np.float32)
        else:
            label = label.astype(np.float32)
            
        if len(label.shape) == 3:
            label = label[..., 0]
        
        return image, label, img_path.name

def show_random_samples(dataset, num_samples=10):
    """Показывает случайные примеры из датасета"""
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Создаем сетку: 2 колонки (изображение + маска) на каждый пример
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    # Если только один пример, axes будет не 2D массивом
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        image, label, filename = dataset[idx]
        
        # Показ изображения
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis('off')
        
        # Показ маски
        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title("Label (Roads)")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_path = "data/tiff"
    
    train_dataset = RoadSegmentationDataset(
        images_dir=os.path.join(base_path, "train"),
        labels_dir=os.path.join(base_path, "train_labels")
    )
    
    test_dataset = RoadSegmentationDataset(
        images_dir=os.path.join(base_path, "test"),
        labels_dir=os.path.join(base_path, "test_labels")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} изображений")
    print(f"Test: {len(test_dataset)} изображений")
    
    print("Выводим 10 случайных примеров...")
    show_random_samples(train_dataset, num_samples=10)
    
    print("\nТест DataLoader:")
    for images, labels, filenames in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break