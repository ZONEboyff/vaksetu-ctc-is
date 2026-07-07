"""
dataset_v2.py — Dataset with normalization and augmentation for the v2 pipeline.

Same data loading as dataset.py, but applies:
  1. normalize.py  → shoulder-relative normalization (always)
  2. augment.py    → random augmentations (training only)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import (
    SIGNS,
    NONE_DIR,
    BATCH_SIZE,
    VAL_SPLIT,
    SEED,
)
from normalize import normalize_sequence
from augment import augment


class SignDatasetV2(Dataset):
    """Loads samples with normalization + optional augmentation."""

    def __init__(self, samples, training=False):
        """
        Parameters
        ----------
        samples  : list of (file_path: str, label: torch.LongTensor)
        training : bool — if True, apply augmentations
        """
        self.samples  = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)   # (T, 288)

        # 1. Always normalize
        data = normalize_sequence(data)

        # 2. Augment only during training
        if self.training:
            data = augment(data)

        seq = torch.from_numpy(data.copy())   # (T', 288) — T' may differ after augmentation
        return seq, label.clone()


def ctc_collate_fn(batch):
    """
    Custom collate for CTC:
      - Pads sequences to the same T_max
      - Concatenates labels into a 1-D tensor
    """
    sequences, labels = zip(*batch)

    padded_seqs   = pad_sequence(sequences, batch_first=False)   # (T_max, N, 288)
    input_lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

    non_empty = [l for l in labels if len(l) > 0]
    targets   = torch.cat(non_empty) if non_empty else torch.tensor([], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return padded_seqs, targets, input_lengths, target_lengths


def _collect_files(directory, label_tensor):
    """Return list of (path, label_tensor) for all sample_*.npy in directory."""
    if not os.path.isdir(directory):
        return []
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("sample_") and f.endswith(".npy")
    )
    return [(p, label_tensor) for p in files]


def build_dataloaders_v2(signs=None, none_dir=NONE_DIR, batch_size=BATCH_SIZE):
    """
    Build train/val DataLoaders with normalization and augmentation.

    Returns
    -------
    train_loader, val_loader
    """
    if signs is None:
        signs = SIGNS

    all_samples = []

    # ── Sign samples ──────────────────────────────────────────────────────
    for sign_key, sign_cfg in signs.items():
        label_tensor = torch.tensor(sign_cfg["labels"], dtype=torch.long)
        samples      = _collect_files(sign_cfg["data_dir"], label_tensor)
        print(f"[DatasetV2] {sign_key:20s}  samples={len(samples):4d}  "
              f"labels={sign_cfg['labels']}")
        if len(samples) == 0:
            print(f"            WARNING: No samples in '{sign_cfg['data_dir']}'.")
        all_samples.extend(samples)

    if len(all_samples) == 0:
        raise FileNotFoundError(
            "No sign samples found. Run collect_data.py first."
        )

    # ── None / idle samples ───────────────────────────────────────────────
    none_label   = torch.tensor([], dtype=torch.long)
    none_samples = _collect_files(none_dir, none_label)
    print(f"[DatasetV2] {'none':20s}  samples={len(none_samples):4d}  labels=[]")
    all_samples.extend(none_samples)

    # ── Shuffle & split ───────────────────────────────────────────────────
    rng     = np.random.default_rng(SEED)
    indices = list(rng.permutation(len(all_samples)))
    all_samples = [all_samples[i] for i in indices]

    n_val         = max(1, int(len(all_samples) * VAL_SPLIT))
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    n_sign_train = sum(1 for _, l in train_samples if len(l) > 0)
    n_none_train = sum(1 for _, l in train_samples if len(l) == 0)
    n_sign_val   = sum(1 for _, l in val_samples   if len(l) > 0)
    n_none_val   = sum(1 for _, l in val_samples   if len(l) == 0)

    print(f"[DatasetV2] Total={len(all_samples)}")
    print(f"            Train={len(train_samples)} "
          f"(sign={n_sign_train}, none={n_none_train})  "
          f"Val={len(val_samples)} "
          f"(sign={n_sign_val}, none={n_none_val})")
    print(f"[DatasetV2] Normalization: ON   |   Augmentation: TRAIN only")

    train_ds = SignDatasetV2(train_samples, training=True)
    val_ds   = SignDatasetV2(val_samples,   training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_samples)),
        shuffle=True,
        collate_fn=ctc_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_samples)),
        shuffle=False,
        collate_fn=ctc_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
