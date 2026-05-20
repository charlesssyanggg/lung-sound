"""
dataset.py — ICBHI 2017 & SPRSound Data Loading

Handles:
- Parsing ICBHI annotation (.txt) files
- Audio loading and resampling
- Mel spectrogram extraction
- Patient-wise train/test split (no data leakage)
- SPRSound loading for cross-dataset evaluation
"""

import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

# ── Label mapping ──────────────────────────────────────────
ICBHI_LABELS = {
    (False, False): 0,   # Normal
    (True,  False): 1,   # Crackle only
    (False, True ): 2,   # Wheeze only
    (True,  True ): 3,   # Both
}
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


# ══════════════════════════════════════════════════════════
# ICBHI 2017 Dataset
# ══════════════════════════════════════════════════════════

def parse_icbhi_annotation(txt_path: str) -> list:
    """
    Parse a single ICBHI annotation file.

    Format per line:
        start_time  end_time  crackle(0/1)  wheeze(0/1)

    Returns:
        List of (start, end, label_idx) tuples.
    """
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start   = float(parts[0])
            end     = float(parts[1])
            crackle = int(parts[2]) == 1
            wheeze  = int(parts[3]) == 1
            label   = ICBHI_LABELS[(crackle, wheeze)]
            cycles.append((start, end, label))
    return cycles


def extract_mel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Convert raw audio to log-Mel spectrogram.

    Args:
        audio: 1-D float32 array
        sr:    sample rate
        cfg:   config dict with n_mels, n_fft, hop_length

    Returns:
        np.ndarray of shape (n_mels, T)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=cfg.get("n_mels", 128),
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def resize_spec(spec: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize spectrogram to (size × size) via simple interpolation."""
    from PIL import Image
    img = Image.fromarray(spec).resize((size, size), Image.BILINEAR)
    return np.array(img)


class ICBHIDataset(Dataset):
    """
    ICBHI 2017 Respiratory Sound Dataset.

    Args:
        root:       Path to directory containing .wav and .txt files.
        patient_ids: List of patient IDs to include (for train/test split).
        cfg:        Config dict.
        augment:    If True, apply SpecAugment.

    Directory structure expected:
        root/
        ├── 101_1b1_Al_sc_Meditron.wav
        ├── 101_1b1_Al_sc_Meditron.txt
        └── ...
    """

    def __init__(self, root: str, patient_ids: list,
                 cfg: dict, augment: bool = False):
        self.cfg     = cfg
        self.augment = augment
        self.sr      = cfg.get("sample_rate", 22050)
        self.dur     = cfg.get("duration", 5)
        self.target_len = self.sr * self.dur

        self.samples = []   # List of (wav_path, start, end, label)
        self._load(root, patient_ids)

        print(f"[ICBHIDataset] {len(self.samples)} cycles | "
              f"augment={augment}")
        self._print_distribution()

    def _load(self, root: str, patient_ids: list):
        txt_files = glob.glob(os.path.join(root, "*.txt"))
        for txt_path in txt_files:
            # Extract patient ID from filename (first token before '_')
            fname   = os.path.basename(txt_path)
            pat_id  = int(fname.split("_")[0])
            if pat_id not in patient_ids:
                continue
            wav_path = txt_path.replace(".txt", ".wav")
            if not os.path.exists(wav_path):
                continue
            for start, end, label in parse_icbhi_annotation(txt_path):
                self.samples.append((wav_path, start, end, label))

    def _print_distribution(self):
        counts = [0] * 4
        for *_, label in self.samples:
            counts[label] += 1
        total = len(self.samples)
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:8s}: {counts[i]:4d} ({counts[i]/total*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, start, end, label = self.samples[idx]

        # Load audio segment
        audio, _ = librosa.load(
            wav_path, sr=self.sr,
            offset=start, duration=(end - start)
        )

        # Pad or trim to fixed length
        if len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        else:
            audio = audio[:self.target_len]

        # Mel spectrogram → (128, 128)
        spec = extract_mel(audio, self.sr, self.cfg)
        spec = resize_spec(spec, size=128)

        # Normalize to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)

        # SpecAugment (training only)
        if self.augment:
            x = self._spec_augment(x)

        return x, label

    @staticmethod
    def _spec_augment(x: torch.Tensor,
                      time_mask_ratio: float = 0.25) -> torch.Tensor:
        """Apply time masking (SpecAugment). Frequency masking disabled."""
        x = x.clone()
        T = x.shape[-1]
        t = int(T * time_mask_ratio * np.random.rand())
        t0 = np.random.randint(0, max(1, T - t))
        x[:, :, t0:t0 + t] = 0.0
        return x


def get_patient_split(root: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Patient-wise train/test split.
    Avoids same patient appearing in both train and test sets.

    Returns:
        train_ids, test_ids: sorted lists of patient IDs
    """
    txt_files   = glob.glob(os.path.join(root, "*.txt"))
    patient_ids = sorted(set(
        int(os.path.basename(f).split("_")[0]) for f in txt_files
    ))
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids := np.array(patient_ids))
    n_train = int(len(patient_ids) * train_ratio)
    return list(patient_ids[:n_train]), list(patient_ids[n_train:])


def get_icbhi_loaders(root: str, cfg: dict,
                      batch_size: int = 64) -> tuple:
    """
    Build ICBHI train and test DataLoaders.

    Args:
        root:       Path to ICBHI audio_and_txt_files directory.
        cfg:        Config dict.
        batch_size: Batch size.

    Returns:
        (train_loader, test_loader)
    """
    train_ids, test_ids = get_patient_split(
        root,
        train_ratio=cfg.get("train_ratio", 0.8),
        seed=cfg.get("seed", 42)
    )
    print(f"[Split] Train patients: {len(train_ids)} | "
          f"Test patients: {len(test_ids)}")

    train_ds = ICBHIDataset(root, train_ids, cfg, augment=True)
    test_ds  = ICBHIDataset(root, test_ids,  cfg, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════
# SPRSound Dataset (cross-dataset evaluation)
# ══════════════════════════════════════════════════════════

class SPRSoundDataset(Dataset):
    """
    SPRSound Paediatric Respiratory Sound Database.
    Used ONLY for cross-dataset evaluation (never for training).

    SPRSound has 7 classes; we map to ICBHI's 4 classes:
        normal              → 0 (Normal)
        crackle / fine crackle → 1 (Crackle)
        wheeze / rhonchus   → 2 (Wheeze)
        wheeze+crackle      → 3 (Both)

    Adjust `SPR_LABEL_MAP` to match the version of SPRSound you download.
    """

    SPR_LABEL_MAP = {
        "normal":       0,
        "crackle":      1,
        "fine_crackle": 1,
        "wheeze":       2,
        "rhonchus":     2,
        "wheeze_crackle": 3,
        "stridor":      2,   # mapped to wheeze (closest)
    }

    def __init__(self, root: str, cfg: dict):
        self.cfg    = cfg
        self.sr     = cfg.get("sample_rate", 22050)
        self.dur    = cfg.get("duration", 5)
        self.target_len = self.sr * self.dur
        self.samples = []
        self._load(root)
        print(f"[SPRSoundDataset] {len(self.samples)} samples loaded")

    def _load(self, root: str):
        """
        Expected structure:
            root/
            ├── audio/    (*.wav files)
            └── labels/   (*.txt or *.csv with label per file)
        Adapt this method to match your actual SPRSound directory layout.
        """
        audio_dir = os.path.join(root, "audio")
        label_dir = os.path.join(root, "labels")
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(
                f"SPRSound audio directory not found: {audio_dir}\n"
                "Please download SPRSound from: "
                "https://github.com/SPRSound/SPRSound"
            )
        for wav_file in glob.glob(os.path.join(audio_dir, "*.wav")):
            fname = os.path.splitext(os.path.basename(wav_file))[0]
            lbl_file = os.path.join(label_dir, fname + ".txt")
            if not os.path.exists(lbl_file):
                continue
            with open(lbl_file) as f:
                raw_label = f.read().strip().lower().replace(" ", "_")
            label = self.SPR_LABEL_MAP.get(raw_label)
            if label is None:
                continue
            self.samples.append((wav_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        audio, _ = librosa.load(wav_path, sr=self.sr, duration=self.dur)
        if len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        else:
            audio = audio[:self.target_len]
        spec = extract_mel(audio, self.sr, self.cfg)
        spec = resize_spec(spec, size=128)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return x, label


def get_sprout_loader(root: str, cfg: dict,
                      batch_size: int = 64) -> DataLoader:
    ds = SPRSoundDataset(root, cfg)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=2)


# ══════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)["data"]

    root = cfg["icbhi_root"]
    if not os.path.exists(root):
        print(f"[WARNING] ICBHI data not found at: {root}")
        print("Please download the dataset first.")
        print("See README.md → Quick Start → Download ICBHI 2017")
    else:
        train_loader, test_loader = get_icbhi_loaders(root, cfg)
        x, y = next(iter(train_loader))
        print(f"Batch: x={x.shape}, y={y.shape}, labels={y[:8].tolist()}")
