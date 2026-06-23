import os
import glob
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from torch.optim.lr_scheduler import LinearLR


def get_filtered_onsets(y, sr, min_dist_sec=0.3):
    import librosa

    raw_onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="samples")
    filtered = []
    min_dist = sr * min_dist_sec
    for o in raw_onsets:
        if not filtered or (o - filtered[-1]) > min_dist:
            filtered.append(o)
    return filtered


# ---------------------------------------------------------
# 1. CoAtNet Architecture
# ---------------------------------------------------------
class MBConv(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MBConv, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class CoAtNet(nn.Module):
    """
    Two Depth-wise Convolutional layers (MBConv)
    Two global relative attention layers (Transformer)
    2D Avg Pool -> Fully Connected
    """

    def __init__(self, num_classes=36):
        super(CoAtNet, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Depth-wise Convolutional Phase
        self.mbconv1 = MBConv(32, 64, stride=2)
        self.mbconv2 = MBConv(64, 128, stride=2)

        # Self-Attention Phase
        self.d_model = 128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.attention1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention2 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Pooling & Classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)

        # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        x_attn = self.attention1(x_flat)
        x_attn = self.attention2(x_attn)

        # (B, H*W, C) -> (B, C, H, W)
        x_attn = x_attn.permute(0, 2, 1).view(B, C, H, W)

        x_pool = self.pool(x_attn)
        x_pool = torch.flatten(x_pool, 1)
        out = self.fc(x_pool)
        return out


# ---------------------------------------------------------
# 2. Data Processing & Augmentation (Dataset)
# ---------------------------------------------------------
class KeystrokeDataset(Dataset):
    def __init__(self, data_list, is_train=True):
        self.samples = []
        self.labels = []
        self.is_train = is_train

        # Max Mask Percentage 0.1 on 64x64 means max_mask=6.4 -> 6
        # Number of Masks Per Axis 2
        self.freq_mask1 = T.FrequencyMasking(freq_mask_param=6)
        self.freq_mask2 = T.FrequencyMasking(freq_mask_param=6)
        self.time_mask1 = T.TimeMasking(time_mask_param=6)
        self.time_mask2 = T.TimeMasking(time_mask_param=6)

        target_len = 14400  # 14400 samples / 225 hop_length = 64 frames (64x64 image)

        for chunk, label in data_list:
            if is_train:
                # Base + 2 randomly time-shifted augmentations (+/- 40%)
                rates = [1.0, np.random.uniform(0.6, 1.4), np.random.uniform(0.6, 1.4)]
            else:
                rates = [1.0]

            for rate in rates:
                if rate != 1.0:
                    y_stretched = librosa.effects.time_stretch(y=chunk, rate=rate)
                else:
                    y_stretched = chunk

                # Pad or crop to exactly 14400 samples
                if len(y_stretched) > target_len:
                    y_stretched = y_stretched[:target_len]
                else:
                    y_stretched = np.pad(
                        y_stretched, (0, target_len - len(y_stretched))
                    )

                melspec = librosa.feature.melspectrogram(
                    y=y_stretched, sr=44100, n_fft=1024, hop_length=225, n_mels=64
                )
                melspec_db = librosa.power_to_db(melspec, ref=np.max)

                # Normalise to 0-1
                melspec_db = (melspec_db - melspec_db.min()) / (
                    melspec_db.max() - melspec_db.min() + 1e-6
                )

                self.samples.append(melspec_db)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)

        # SpecAugment during training
        if self.is_train:
            x = self.freq_mask1(x)
            x = self.freq_mask2(x)
            x = self.time_mask1(x)
            x = self.time_mask2(x)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def load_all_chunks(data_dir):
    classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    all_data = []

    # Approx 0.326 seconds for exactly 14400 samples at 44100Hz
    chunk_samples = 14400

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for audio_file in glob.glob(os.path.join(cls_dir, "*.wav")):
            y, sr = librosa.load(audio_file, sr=44100)
            onsets = get_filtered_onsets(y, sr)

            for onset in onsets:
                start = int(onset)
                end = start + chunk_samples
                if end > len(y):
                    continue
                all_data.append((y[start:end], idx))

    return all_data, classes


# ---------------------------------------------------------
# 3. Training Loop (Hyperparameters matching paper exactly)
# ---------------------------------------------------------
def train_model(data_dir):
    print("Extracting keystrokes and building dataset...")
    all_data, classes = load_all_chunks(data_dir)

    if not all_data:
        print("No training data found!")
        return None

    # Random Data Split (80% train, 20% val)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(
        f"Applying time-shift (+/- 40%) data augmentation to {len(train_data)} training chunks..."
    )
    train_dataset = KeystrokeDataset(train_data, is_train=True)
    val_dataset = KeystrokeDataset(val_data, is_train=False)

    # 16 Batch Size
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using compute device: {device}")

    model = CoAtNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Max LR = 5e-4

    # Epochs 1100, Linear Annealing Schedule
    epochs = 1100
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = 50

    print(f"\n--- Hyperparameters ---")
    print(
        f"Classes: {len(classes)} | Epochs: {epochs} | Batch: 16 | Split: Random (80/20)"
    )
    print(f"Max LR: 5e-4 | Annealing: Linear | FFT: 1024 | Hop: 225 | Mels: 64")
    print(f"Augmentations: 40% Timeshift, 10% Spec Masking (2x axis)")
    print(f"Train Dataset Size (Augmented): {len(train_dataset)}")
    print(f"Val Dataset Size (Base): {len(val_dataset)}")
    print(f"-----------------------\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "keystroke_model_best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if True:
            print(
                f"Epoch [{epoch + 1:4d}/{epochs}] Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Peak Val: {best_val_acc:.4f}"
            )

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered after {epoch + 1} epochs. No improvement for {early_stopping_patience} epochs."
            )
            break

    print(f"\nTraining Complete! Peak Validation Accuracy: {best_val_acc:.4f}")

    # Save classes
    with open("classes.txt", "w") as f:
        f.write("\n".join(classes))

    return model


def predict_audio(
    audio_path, model_path="keystroke_model_best.pth", classes_path="classes.txt"
):
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("Model or classes file not found. Please train first.")
        return

    with open(classes_path, "r") as f:
        classes = f.read().splitlines()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = CoAtNet(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y, sr = librosa.load(audio_path, sr=44100)
    onsets = get_filtered_onsets(y, sr)

    predictions = []
    target_len = 14400  # 64x64

    with torch.no_grad():
        for onset in onsets:
            start = int(onset)
            end = start + target_len
            if end > len(y):
                chunk = np.pad(y[int(start) :], (0, end - len(y)))
            else:
                chunk = y[int(start) : int(end)]

            melspec = librosa.feature.melspectrogram(
                y=chunk, sr=44100, n_fft=1024, hop_length=225, n_mels=64
            )
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            melspec_db = (melspec_db - melspec_db.min()) / (
                melspec_db.max() - melspec_db.min() + 1e-6
            )

            x = (
                torch.tensor(melspec_db, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(classes[int(predicted.item())])

    text = "".join(predictions)
    print(f"Predicted Keystrokes: {text}")
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Acoustic Keystroke Attack (hacker de teclado)"
    )
    parser.add_argument(
        "--treinar", type=str, help="Diretório contendo arquivos de áudio"
    )
    parser.add_argument(
        "--prever", type=str, help="Arquivo de áudio para prever as teclas"
    )
    args = parser.parse_args()

    if args.treinar:
        train_model(args.treinar)
    elif args.prever:
        predict_audio(args.prever)
    else:
        print(
            "Por favor, especifique --treinar <dir_dados> ou --prever <arquivo_audio>"
        )
