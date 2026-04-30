import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tifffile as tiff

# ------------------ DATASET ------------------
class RoadDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        print(f"[DEBUG] Initializing dataset...")
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgs = sorted(list(self.img_dir.glob("*.tiff")))
        print(f"[DEBUG] Found {len(self.imgs)} images in {img_dir}")

    def __len__(self): 
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / img_path.name.replace(".tiff", ".tif")
        
        img = tiff.imread(img_path).astype(np.float32) / 255.0
        lbl = (tiff.imread(lbl_path) > 0).astype(np.float32)
        
        if img.ndim == 2: img = np.expand_dims(img, 0)
        else: img = img.transpose(2, 0, 1)
        lbl = np.expand_dims(lbl, 0)
        
        return torch.tensor(img), torch.tensor(lbl)

# ------------------ FCN HOURGLASS MODEL ------------------
class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        diff_y = e4.size(2) - d4.size(2)
        diff_x = e4.size(3) - d4.size(3)
        e4_cropped = e4[:, :, diff_y//2:diff_y//2 + d4.size(2), diff_x//2:diff_x//2 + d4.size(3)]
        d4 = self.dec4(torch.cat([d4, e4_cropped], 1))

        d3 = self.up3(d4)
        diff_y = e3.size(2) - d3.size(2)
        diff_x = e3.size(3) - d3.size(3)
        e3_cropped = e3[:, :, diff_y//2:diff_y//2 + d3.size(2), diff_x//2:diff_x//2 + d3.size(3)]
        d3 = self.dec3(torch.cat([d3, e3_cropped], 1))

        d2 = self.up2(d3)
        diff_y = e2.size(2) - d2.size(2)
        diff_x = e2.size(3) - d2.size(3)
        e2_cropped = e2[:, :, diff_y//2:diff_y//2 + d2.size(2), diff_x//2:diff_x//2 + d2.size(3)]
        d2 = self.dec2(torch.cat([d2, e2_cropped], 1))

        d1 = self.up1(d2)
        diff_y = e1.size(2) - d1.size(2)
        diff_x = e1.size(3) - d1.size(3)
        e1_cropped = e1[:, :, diff_y//2:diff_y//2 + d1.size(2), diff_x//2:diff_x//2 + d1.size(3)]
        d1 = self.dec1(torch.cat([d1, e1_cropped], 1))
        
        return self.final(d1)

# ------------------ TRAINING & VISUALIZATION ------------------
def run_pipeline(model, train_loader, device, epochs=3):
    print(f"[DEBUG] Starting training pipeline...")
    model.to(device)
    print(f"[DEBUG] Model moved to {device}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(f"[DEBUG] Optimizer and criterion initialized")

    for epoch in range(epochs):
        print(f"\n[DEBUG] ===== EPOCH {epoch+1}/{epochs} START =====")
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            print(f"[DEBUG] Epoch {epoch+1} - Processing batch {batch_idx+1}/{len(train_loader)}")
            print(f"[DEBUG]   Image shape: {imgs.shape}, Label shape: {lbls.shape}")
            
            imgs, lbls = imgs.to(device), lbls.to(device)
            print(f"[DEBUG]   Data moved to device")
            
            optimizer.zero_grad()
            print(f"[DEBUG]   Running forward pass...")
            
            preds = model(imgs)
            print(f"[DEBUG]   Forward pass done. Preds shape: {preds.shape}")
            
            if preds.shape != lbls.shape:
                print(f"[DEBUG]   Cropping labels from {lbls.shape} to {preds.shape}")
                lbls = lbls[:, :, :preds.shape[2], :preds.shape[3]]
            
            print(f"[DEBUG]   Computing loss...")
            loss = criterion(preds, lbls)
            print(f"[DEBUG]   Loss value: {loss.item():.4f}")
            
            print(f"[DEBUG]   Backward pass...")
            loss.backward()
            
            print(f"[DEBUG]   Optimizer step...")
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            print(f"[DEBUG]   Batch {batch_idx+1} completed\n")
            
            # Можно раскомментировать для пропуска подробного вывода после первого батча
            # if batch_idx > 0:
            #     break
        
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"\n[DEBUG] ===== EPOCH {epoch+1}/{epochs} COMPLETE =====")
        print(f"[DEBUG] Average Loss: {avg_loss:.4f}\n")

    # Visualization
    print("[DEBUG] Generating visualization...")
    model.eval()
    imgs, lbls = next(iter(train_loader))
    imgs, lbls = imgs[:4].to(device), lbls[:4].to(device)
    
    with torch.no_grad():
        preds = model(imgs)
        preds = (torch.sigmoid(preds) > 0.5).float()

    fig, axes = plt.subplots(4, 3, figsize=(10, 8))
    for i in range(4):
        img_np = imgs[i].cpu().permute(1,2,0).numpy()
        axes[i,0].imshow(img_np)
        lbl_vis = lbls[i].cpu().squeeze()
        pred_vis = preds[i].cpu().squeeze()
        axes[i,1].imshow(lbl_vis, cmap='gray')
        axes[i,2].imshow(pred_vis, cmap='gray')
        for ax in axes[i]: ax.axis('off')
    axes[0,0].set_title('Input'); axes[0,1].set_title('True Mask'); axes[0,2].set_title('Prediction')
    plt.tight_layout()
    plt.show()
    print("[DEBUG] Visualization complete!")

if __name__ == "__main__":
    print("[DEBUG] Script started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    
    print("[DEBUG] Creating datasets...")
    train_ds = RoadDataset("data/tiff/train", "data/tiff/train_labels")
    print(f"[DEBUG] Train dataset size: {len(train_ds)}")
    
    print("[DEBUG] Creating DataLoader...")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    print(f"[DEBUG] DataLoader created. Batches per epoch: {len(train_loader)}")
    
    print("[DEBUG] Creating model...")
    model = FCN(in_channels=3)
    print("[DEBUG] Model created")
    
    print("[DEBUG] Starting pipeline...")
    run_pipeline(model, train_loader, device, epochs=3)
    print("[DEBUG] Done!")