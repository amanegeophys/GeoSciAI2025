import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model2 import UNet2D

# === Global Settings ===
SAMPLE_RATE = 100
WIN_LENGTH = 30
HOP_LENGTH = 15
N_FFT = 60
DURATION = 30 * SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_istft(spec_tensor):
    """
    Convert a real/im magnitude spectrogram to waveform via ISTFT.
    spec_tensor: Tensor of shape [batch, 2, freq, time], where spec_tensor[:,0] is real,
    spec_tensor[:,1] is imag.
    """
    # Build complex spectrogram
    complex_spec = spec_tensor[:, 0] + 1j * spec_tensor[:, 1]
    # Hann window on correct device
    window = torch.hann_window(WIN_LENGTH, device=complex_spec.device)
    # Inverse STFT
    wave = torch.istft(
        complex_spec,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window=window,
        normalized=False,
        length=DURATION,
    )
    return wave


def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std.item() == 0:
        std = torch.tensor(1.0, device=tensor.device)
    return (tensor - mean) / std


# === Dataset ===
class MaskDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        spec = data["spec"].float()  # [2, F, T]
        mask = data["mask"].float()
        # Ensure mask has channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [1, F, T]
        # Normalize each channel
        spec_norm = torch.stack(
            [normalize(spec[0]), normalize(spec[1])], dim=0
        )

        pidx = data.get("pidx")
        sidx = data.get("sidx")
        return spec, spec_norm, mask, pidx, sidx


# === Custom Loss (batch-aware) ===
class CustomLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _corrcoef_batch(x, y, eps=1e-8):
        # x, y: [B, L]
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        xm = x - x_mean
        ym = y - y_mean
        corr = (xm * ym).mean(dim=1) / (x.std(dim=1) * y.std(dim=1) + eps)
        return corr

    def forward(self, output, mask, spec, pidx, sidx):
        # output: [B,1,F,T], mask: [B,1,F,T], spec:[B,2,F,T], pidx:[B], sidx:[B]
        batch_size = output.size(0)
        mse = self.mse_loss(output, mask)

        # Apply predicted mask
        masked_spec = spec * output  # [B, 2, F, T]

        # ISTFT to waveform: [B, L]
        denoised_wave = calculate_istft(masked_spec)  # [B, L]
        orig_wave = calculate_istft(spec)

        # 各区間（500サンプル）をまとめて抽出
        pad = 500
        pi = pidx.long()
        si = sidx.long()

        # Index matrix [B, 500]
        idx = torch.arange(-pad, pad, device=spec.device).unsqueeze(
            0
        )  # [1, 1000]
        dn_P = denoised_wave[
            torch.arange(batch_size).unsqueeze(1),
            pi.unsqueeze(1) + idx[:, :pad],
        ]  # [B, 500]
        dn_S = denoised_wave[
            torch.arange(batch_size).unsqueeze(1),
            si.unsqueeze(1) + idx[:, :pad],
        ]  # [B, 500]
        dn_N = denoised_wave[
            torch.arange(batch_size).unsqueeze(1),
            pi.unsqueeze(1) + idx[:, :-pad],
        ]  # [B, 500]

        or_P = orig_wave[
            torch.arange(batch_size).unsqueeze(1),
            pi.unsqueeze(1) + idx[:, :pad],
        ]
        or_S = orig_wave[
            torch.arange(batch_size).unsqueeze(1),
            si.unsqueeze(1) + idx[:, :pad],
        ]
        or_N = orig_wave[
            torch.arange(batch_size).unsqueeze(1),
            pi.unsqueeze(1) + idx[:, :-pad],
        ]

        eps = 1e-8

        # PSNR, SSNR（バッチ全体で）
        psnr = torch.log10(dn_P.std(dim=1) / (dn_N.std(dim=1) + eps))
        ssnr = torch.log10(dn_S.std(dim=1) / (dn_N.std(dim=1) + eps))

        # 相関係数もバッチで
        rhoP = self._corrcoef_batch(dn_P, or_P)
        rhoS = self._corrcoef_batch(dn_S, or_S)
        rhoN = self._corrcoef_batch(dn_N, or_N)

        # 平均
        psnr_avg = psnr.mean()
        ssnr_avg = ssnr.mean()
        rhoP_avg = rhoP.mean()
        rhoS_avg = rhoS.mean()
        rhoN_avg = rhoN.mean()

        # Combine losses
        loss = (
            mse
            - self.alpha * (psnr_avg + ssnr_avg)
            - self.beta * (rhoP_avg + rhoS_avg + rhoN_avg)
        )
        return loss


# === Visualization Function ===
def save_validation_visualization(
    spec, pred_mask, true_mask, pidx, sidx, output_dir, idx, epoch
):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    # Magnitude of input spec
    spec_mag = torch.abs(spec[:, 0] + spec[:, 1] * 1j)

    # Handle mask dimensions
    true_mask_viz = true_mask.squeeze(1)[0].cpu()
    pred_mask_viz = pred_mask.detach().squeeze(1)[0].cpu()

    axes[0].imshow(
        spec_mag[0].cpu(), extent=[0, 30, 0, 50], aspect="auto", origin="lower"
    )
    axes[0].vlines(
        pidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="r", lw=1
    )
    axes[0].vlines(
        sidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="b", lw=1
    )
    axes[0].set_title("Input |Spec|")
    axes[1].imshow(
        true_mask_viz,
        extent=[0, 30, 0, 50],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1].vlines(
        pidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="r", lw=1
    )
    axes[1].vlines(
        sidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="b", lw=1
    )
    axes[1].set_title("True Mask")
    axes[2].imshow(
        pred_mask_viz,
        extent=[0, 30, 0, 50],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[2].vlines(
        pidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="r", lw=1
    )
    axes[2].vlines(
        sidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="b", lw=1
    )
    axes[2].set_title("Predicted Mask")
    axes[3].imshow(
        (spec_mag[0].cpu() * pred_mask_viz).cpu(),
        extent=[0, 30, 0, 50],
        aspect="auto",
        origin="lower",
    )
    axes[3].vlines(
        pidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="r", lw=1
    )
    axes[3].vlines(
        sidx[0].cpu() / HOP_LENGTH * (30 / 201), 0, 50, color="b", lw=1
    )
    axes[3].set_title("Filtered |Spec|")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"epoch{epoch}_val_result_{idx:02d}.png")
    )
    plt.close()


# === Train Function ===
def train_model(train_loader, val_loader, epochs=50, output_dir="output"):
    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CustomLoss(alpha=0.002, beta=0.01)
    best_val_loss = float("inf")
    patience, counter = 4, 0

    os.makedirs(output_dir, exist_ok=True)
    model_dir = Path(output_dir, "model")
    model_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        for spec, spec_norm, mask, pidx, sidx in tqdm(
            train_loader, desc=f"Epoch {epoch:03} Training"
        ):
            spec, spec_norm, mask = (
                spec.to(device),
                spec_norm.to(device),
                mask.to(device),
            )
            pidx = pidx.to(device)
            sidx = sidx.to(device)

            output = model(spec_norm)
            loss = criterion(output, mask, spec, pidx, sidx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)
        print(f"Epoch {epoch:03} Train Loss: {avg_train:.6f}")

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for i, (spec, spec_norm, mask, pidx, sidx) in enumerate(
                val_loader
            ):
                spec, spec_norm, mask = (
                    spec.to(device),
                    spec_norm.to(device),
                    mask.to(device),
                )
                pidx = pidx.to(device)
                sidx = sidx.to(device)
                output = model(spec_norm)
                loss = criterion(output, mask, spec, pidx, sidx)
                total_val += loss.item()

                if i < 5:
                    save_validation_visualization(
                        spec,
                        output,
                        mask,
                        pidx,
                        sidx,
                        os.path.join(output_dir, "viz"),
                        i,
                        epoch,
                    )

        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)
        print(f"Epoch {epoch:03} Val Loss: {avg_val:.6f}")

        torch.save(model.state_dict(), model_dir / f"{epoch}.pt")
        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            print("✅ Saved best model.")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Save loss log
    df = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    df.to_csv(os.path.join(output_dir, "loss_log.csv"), index=False)


# === Main ===
if __name__ == "__main__":
    train_ds = MaskDataset("unet_snr/datasets/train")
    val_ds = MaskDataset("unet_snr/datasets/val")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    train_model(
        train_loader,
        val_loader,
        epochs=500,
        output_dir="unet_snr",
    )
