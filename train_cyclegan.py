# Required Libraries
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import ShortTimeFFT
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SAMPLE_RATE = 100
WIN_LENGTH = 30
HOP_LENGTH = 15
N_FFT = 60
DURATION = 30 * SAMPLE_RATE  # 90 seconds
# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy ShortTimeFFT (Replace with actual implementation)

SFT = ShortTimeFFT(
    win=np.hanning(WIN_LENGTH),
    hop=HOP_LENGTH,
    fs=SAMPLE_RATE,
    fft_mode="onesided2X",
    mfft=N_FFT,
    scale_to="magnitude",
)

# Dataset Class
torch.manual_seed(0)


class SeismicDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        wave = data["wave"].numpy()
        clean = data["denoised"].numpy()

        return (
            torch.from_numpy(wave).float(),
            torch.from_numpy(clean).float(),
        )


# === STFT utility ===
def compute_stft(x):
    B, C, T = x.shape
    x = x.view(B * C, T)
    stft_out = torch.stft(
        x,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(WIN_LENGTH).to(x.device),
        return_complex=True,
        pad_mode="constant",
        # normalized=True,
    )
    return stft_out.view(B, C, *stft_out.shape[-2:])  # [B, C, F, T]


def compute_istft(X, length):
    B, C, F, T = X.shape
    X = X.view(B * C, F, T)
    x = torch.istft(
        X,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(WIN_LENGTH).to(X.device),
        length=length,
        # normalized=True,
    )
    return x.view(B, C, -1)  # [B, C, T]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大値
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))  # [B,1,H,W]
        return x * attn


class UNet2D(nn.Module):
    """
    2D U-Net (encoder-decoder)
    ・入力サイズが 2 回 MaxPool されても割り切れない場合に備え、
      skip 接続側（エンコーダ）の特徴量をパディング／クロップで
      デコーダ側と揃える。
    """

    def __init__(self, in_channels=3, base_channels=10):
        super().__init__()
        # エンコーダ
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        # ボトルネック
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)

        # デコーダ
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def conv_block(self, in_c, out_c, use_se=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if use_se:
            layers.append(SpatialAttention())
        return nn.Sequential(*layers)

    def forward(self, x):
        # エンコーダ
        e1 = self.enc1(x)  # [B, C, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 2C, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 4C, H/4, W/4]

        # ボトルネック
        b = self.bottleneck(self.pool(e3))  # [B, 8C, H/8, W/8]

        # デコーダ + スキップ接続
        d3 = self.up3(b)
        d3 = self.pad_if_needed(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.pad_if_needed(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.pad_if_needed(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)

        return out

    def pad_if_needed(self, upsampled, encoder_feature):
        # サイズが一致しないときはパディング
        diffY = encoder_feature.size(2) - upsampled.size(2)
        diffX = encoder_feature.size(3) - upsampled.size(3)
        return nn.functional.pad(upsampled, [0, diffX, 0, diffY])


class Discriminator2D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )


    def forward(self, x):
        return torch.sigmoid(self.net(x))

# Save Spectrogram Visualization
def save_spectrogram_plot(
    noisy, pred, clean, noisy_phase, clean_phase, epoch, idx, output_dir
):
    fig, axes = plt.subplots(6, 3, figsize=(12, 10))
    titles = ["UD", "NS", "EW"]
    noisy_complex = torch.expm1(noisy) * torch.exp(1j * noisy_phase)
    noisy_wave = (
        compute_istft(noisy_complex, DURATION)[0].detach().cpu().numpy()
    )
    pred_complex = torch.expm1(pred) * torch.exp(1j * noisy_phase)
    pred_wave = compute_istft(pred_complex, DURATION)[0].detach().cpu().numpy()
    clean_complex = torch.expm1(clean) * torch.exp(1j * clean_phase)
    clean_wave = (
        compute_istft(clean_complex, DURATION)[0].detach().cpu().numpy()
    )
    for i in range(3):
        axes[0, i].set_title(f"Noisy {titles[i]}")
        axes[0, i].plot(noisy_wave[i])
        axes[1, i].imshow(
            noisy[0].cpu().numpy()[i],
            aspect="auto",
            origin="lower",
            extent=[
                0,
                noisy[0].cpu().numpy().shape[1] * SFT.hop / SFT.fs,
                0,
                SFT.fs / 2,
            ],
        )

        axes[2, i].set_title(f"Pred {titles[i]}")
        axes[2, i].plot(pred_wave[i])
        axes[3, i].imshow(
            pred[0].cpu().numpy()[i],
            aspect="auto",
            origin="lower",
            extent=[
                0,
                pred[0].cpu().numpy().shape[1] * SFT.hop / SFT.fs,
                0,
                SFT.fs / 2,
            ],
        )

        axes[4, i].set_title(f"Answer {titles[i]}")
        axes[4, i].plot(clean_wave[i])
        axes[5, i].imshow(
            clean[0].cpu().numpy()[i],
            aspect="auto",
            origin="lower",
            extent=[
                0,
                clean[0].cpu().numpy().shape[1] * SFT.hop / SFT.fs,
                0,
                SFT.fs / 2,
            ],
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/epoch_{epoch:03}_sample_{idx}.png")
    plt.close()


# Training Loop
def train_cyclegan(train_loader, val_loader, epochs=50, output_dir="results"):
    G_NC = UNet2D().to(device)
    G_CN = UNet2D().to(device)
    D_C = Discriminator2D().to(device)
    D_N = Discriminator2D().to(device)

    adversarial_loss = nn.BCELoss()
    cycle_loss = nn.L1Loss()

    optim_G = torch.optim.Adam(
        list(G_NC.parameters()) + list(G_CN.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )
    optim_D = torch.optim.Adam(
        list(D_C.parameters()) + list(D_N.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        G_NC.train()
        G_CN.train()
        D_C.train()
        D_N.train()
        total_loss = 0.0

        for noisy_wave, clean_wave in tqdm(
            train_loader, desc=f"Epoch {epoch:03} Training"
        ):
            with torch.no_grad():
                noisy_stft = compute_stft(noisy_wave)
                clean_stft = compute_stft(clean_wave)
                noisy = torch.log1p(noisy_stft.abs())
                noisy_phase = noisy_stft.angle().to(device)
                clean = torch.log1p(clean_stft.abs())
                clean_phase = clean_stft.angle().to(device)

            noisy = noisy.to(device)
            clean = clean.to(device)

            fake_clean = G_NC(noisy)
            predfake_clean = D_C(fake_clean.detach())
            predreal_clean = D_C(clean)

            loss_D_C = adversarial_loss(
                predfake_clean, torch.zeros_like(predfake_clean)
            ) + adversarial_loss(
                predreal_clean, torch.ones_like(predreal_clean)
            )

            fake_noisy = G_CN(clean)
            predfake_noisy = D_N(fake_noisy.detach())
            predreal_noisy = D_N(noisy)

            loss_D_N = adversarial_loss(
                predfake_noisy, torch.zeros_like(predfake_noisy)
            ) + adversarial_loss(
                predreal_noisy, torch.ones_like(predreal_noisy)
            )

            loss_D = (loss_D_C + loss_D_N) / 2.0
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            rec_noisy = G_CN(fake_clean)
            rec_clean = G_NC(fake_noisy)
            loss_cycle = cycle_loss(rec_noisy, noisy) + cycle_loss(
                rec_clean, clean
            )

            identity_clean = G_NC(clean)
            identity_noisy = G_CN(noisy)
            loss_identity = cycle_loss(identity_clean, clean) + cycle_loss(
                identity_noisy, noisy
            )

            predfake_noisy = D_N(fake_noisy)
            predfake_clean = D_C(fake_clean)
            loss_gan = adversarial_loss(
                predfake_clean, torch.ones_like(predfake_clean)
            ) + adversarial_loss(
                predfake_noisy, torch.ones_like(predfake_noisy)
            )

            # loss_G = loss_gan + 10 * loss_cycle + 5 * loss_identity
            loss_G = loss_gan + 3 * loss_cycle


            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            total_loss += loss_G.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03}, Train Loss: {avg_train_loss:.6f}")

        G_NC.eval()
        G_CN.eval()
        D_C.eval()
        D_N.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (noisy_wave, clean_wave) in enumerate(val_loader):
                noisy_stft = compute_stft(noisy_wave)
                clean_stft = compute_stft(clean_wave)
                noisy = torch.log1p(noisy_stft.abs())
                noisy_phase = noisy_stft.angle().to(device)
                clean = torch.log1p(clean_stft.abs())
                clean_phase = clean_stft.angle().to(device)

                noisy = noisy.to(device)
                clean = clean.to(device)

                fake_clean = G_NC(noisy)
                fake_noisy = G_CN(clean)

                rec_noisy = G_CN(fake_clean)
                rec_clean = G_NC(fake_noisy)
                loss_cycle = cycle_loss(rec_noisy, noisy) + cycle_loss(
                    rec_clean, clean
                )

                identity_clean = G_NC(clean)
                identity_noisy = G_CN(noisy)
                loss_identity = cycle_loss(identity_clean, clean) + cycle_loss(
                    identity_noisy, noisy
                )

                predfake_noisy = D_N(fake_noisy)
                predfake_clean = D_C(fake_clean)
                loss_gan = adversarial_loss(
                    predfake_clean, torch.ones_like(predfake_clean)
                ) + adversarial_loss(
                    predfake_noisy, torch.ones_like(predfake_noisy)
                )

                # loss_G = loss_gan + 10 * loss_cycle + 5 * loss_identity
                loss_G = loss_gan + 3 * loss_cycle

                val_loss += loss_G.item()

                if i < 5:
                    save_spectrogram_plot(
                        noisy,
                        fake_clean,
                        clean,
                        noisy_phase,
                        clean_phase,
                        epoch,
                        i,
                        f"{output_dir}/viz",
                    )

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:03}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(G_NC.state_dict(), f"{output_dir}/best_generator_NC.pt")
            torch.save(G_CN.state_dict(), f"{output_dir}/best_generator_CN.pt")


# Main Execution
if __name__ == "__main__":
    train_ds = SeismicDataset("datasets_mask/train")
    val_ds = SeismicDataset("datasets_mask/val")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_cyclegan(
        train_loader, val_loader, epochs=50, output_dir="cyclegan_output"
    )
