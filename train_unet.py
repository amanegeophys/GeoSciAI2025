import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from model import UNet2D

# === Global Settings ===
SAMPLE_RATE = 100
WIN_LENGTH = 30
HOP_LENGTH = 15
N_FFT = 60
DURATION = 30 * SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
class MaskDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        spec = data["spec"].float()  # [6, F, T]
        mask = data["mask"].float()  # [3, F, T]
        return spec, mask


# === Visualization Function ===
def save_validation_visualization(spec, pred_mask, true_mask, output_dir, idx, epoch):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    titles = ["UD", "NS", "EW"]
    spec_mag = (spec[:, 0::2, :, :]**2 + spec[:, 1::2, :, :]**2).sqrt()

    for i in range(3):
        axes[0, i].imshow(spec_mag[0, i].cpu(), aspect="auto", origin="lower")
        axes[0, i].set_title(f"Input |Spec| {titles[i]}")
        axes[1, i].imshow(true_mask[0, i].cpu(), aspect="auto", origin="lower", vmin=0, vmax=1)
        axes[1, i].set_title(f"True Mask {titles[i]}")
        axes[2, i].imshow(pred_mask[0, i].detach().cpu(), aspect="auto", origin="lower", vmin=0, vmax=1)
        axes[2, i].set_title(f"Predicted Mask {titles[i]}")
        axes[3, i].imshow((spec_mag[0, i] * pred_mask[0, i]).cpu(), aspect="auto", origin="lower")
        axes[3, i].set_title(f"Filtered |Spec| {titles[i]}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch{epoch}_val_result_{idx:02d}.png"))
    plt.close()

# === Train Function ===
def train_model(train_loader, val_loader, epochs=50, output_dir="output"):
    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_val_loss = float('inf')
    patience, counter = 7, 0

    os.makedirs(output_dir, exist_ok=True)
     
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for spec, mask in tqdm(train_loader, desc=f"Epoch {epoch:03} Training"):
            spec, mask = spec.to(device), mask.to(device)
            output = model(spec)
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"Epoch {epoch:03} Train Loss: {avg_train:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (spec, mask) in enumerate(val_loader):
                spec, mask = spec.to(device), mask.to(device)
                output = model(spec)
                loss = criterion(output, mask)
                val_loss += loss.item()

                if i < 5:
                    save_validation_visualization(spec, output, mask, os.path.join(output_dir, "viz"), i, epoch)

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"Epoch {epoch:03} Val Loss: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "model/best_model.pt"))
            print("✅ Saved best model.")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break
    

    df = pd.DataFrame({"epoch": list(range(1, epoch + 1)),
                           "train_loss": train_losses,
                           "val_loss": val_losses})
    df.to_csv(os.path.join(output_dir, "loss_log.csv"), index=False)

# === Main Execution ===
if __name__ == "__main__":
    # 保存済みデータ（拡張含む）を使用
    train_ds = MaskDataset("datasets_mask/train")
    val_ds = MaskDataset("datasets_mask/val")

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    train_model(train_loader, val_loader, epochs=500, output_dir="output_mask")