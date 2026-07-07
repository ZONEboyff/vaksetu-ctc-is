"""
dataset.py — PyTorch Dataset and DataLoader utilities for the CTC pipeline.

Loads from the SIGNS registry in config.py (e.g. good_morning, good_night)
plus data/none/ for negative/idle samples.

torch.nn.CTCLoss expects:
  - log_probs      : (T, N, C)
  - targets        : 1-D concatenated integer labels
  - input_lengths  : (N,)
  - target_lengths : (N,)
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


class SignDataset(Dataset):
    """Loads mixed samples: signed sentences + idle/none."""

    def __init__(self, samples):
        """
        Parameters
        ----------
        samples : list of (file_path: str, label: torch.LongTensor)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)   # (T, 288)
        seq  = torch.from_numpy(data)              # (T, 288)
        return seq, label.clone()


def ctc_collate_fn(batch):
    """
    Custom collate for CTC:
      - Pads sequences to the same T_max
      - Concatenates labels into a 1-D tensor (handles empty labels correctly)
    """
    sequences, labels = zip(*batch)

    padded_seqs   = pad_sequence(sequences, batch_first=False)   # (T_max, N, 258)
    input_lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

    # Handle mix of empty and non-empty labels safely
    non_empty = [l for l in labels if len(l) > 0]
    targets   = torch.cat(non_empty) if non_empty else torch.tensor([], dtype=torch.long)
    # We still track each item's length (0 for none samples)
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


def build_dataloaders(signs=None, none_dir=NONE_DIR):
    """
    Discover all sample files from every sign in the registry plus
    the none/idle directory, split train/val, return DataLoaders.

    Parameters
    ----------
    signs    : dict or None — defaults to config.SIGNS (all registered signs)
    none_dir : str          — path to idle/negative samples
    """
    if signs is None:
        signs = SIGNS

    all_samples = []

    # ── Sign samples ──────────────────────────────────────────────────────
    for sign_key, sign_cfg in signs.items():
        label_tensor = torch.tensor(sign_cfg["labels"], dtype=torch.long)
        samples      = _collect_files(sign_cfg["data_dir"], label_tensor)
        print(f"[Dataset] {sign_key:20s}  samples={len(samples):4d}  "
              f"labels={sign_cfg['labels']}")
        if len(samples) == 0:
            print(f"          WARNING: No samples found in '{sign_cfg['data_dir']}'.")
            print(f"          Run  python collect_data.py --sign {sign_key}  to collect.")
        all_samples.extend(samples)

    if len(all_samples) == 0:
        raise FileNotFoundError(
            "No sign samples found in any directory.\n"
            "Run  python collect_data.py --sign <sign_name>  first."
        )

    # ── None / idle samples ───────────────────────────────────────────────
    none_label   = torch.tensor([], dtype=torch.long)
    none_samples = _collect_files(none_dir, none_label)
    print(f"[Dataset] {'none':20s}  samples={len(none_samples):4d}  labels=[]")
    if len(none_samples) == 0:
        print("[Dataset] WARNING: No 'none' samples found. Training without negatives")
        print("          may cause false positives in inference.")
        print("          Run  python collect_data.py --none  to collect idle samples.")
    all_samples.extend(none_samples)

    # ── Reproducible shuffle & split ──────────────────────────────────────
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

    print(f"[Dataset] Total={len(all_samples)}")
    print(f"          Train={len(train_samples)} "
          f"(sign={n_sign_train}, none={n_none_train})  "
          f"Val={len(val_samples)} "
          f"(sign={n_sign_val}, none={n_none_val})")

    train_ds = SignDataset(train_samples)
    val_ds   = SignDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(BATCH_SIZE, len(train_samples)),
        shuffle=True,
        collate_fn=ctc_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(BATCH_SIZE, len(val_samples)),
        shuffle=False,
        collate_fn=ctc_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader

