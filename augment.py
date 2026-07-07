"""
augment.py — Training-time data augmentations for landmark sequences.

All transforms operate on numpy arrays of shape (T, 288).
They are applied AFTER normalization, ONLY during training.
"""

import numpy as np
from normalize import _POSE_END, _LH_END, _RH_END, _FACE_END


# ─────────────────────────────────────────────────────────────────────────────
# Individual transforms
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(seq, sigma=0.005):
    """Add small Gaussian noise to all coordinate values."""
    noise = np.random.randn(*seq.shape).astype(np.float32) * sigma
    # Don't add noise to visibility channels (every 4th value in the pose block)
    pose_noise = noise[:, :_POSE_END].reshape(-1, 33, 4)
    pose_noise[:, :, 3] = 0.0   # zero out visibility noise
    noise[:, :_POSE_END] = pose_noise.reshape(-1, 33 * 4)
    return seq + noise


def temporal_scale(seq, low=0.8, high=1.2):
    """
    Resample the sequence to simulate speed variation.
    Scale < 1 = faster (fewer frames), scale > 1 = slower (more frames).
    """
    T, D = seq.shape
    factor = np.random.uniform(low, high)
    new_T  = max(4, int(T * factor))

    # Linear interpolation along the time axis
    old_idx = np.linspace(0, T - 1, new_T)
    new_seq = np.zeros((new_T, D), dtype=np.float32)
    for d in range(D):
        new_seq[:, d] = np.interp(old_idx, np.arange(T), seq[:, d])

    return new_seq


def spatial_scale(seq, low=0.9, high=1.1):
    """
    Multiply all spatial coordinates by a random factor.
    Simulates distance/zoom variation.
    """
    factor = np.random.uniform(low, high)
    out    = seq.copy()

    # Scale pose x, y, z (not visibility)
    pose = out[:, :_POSE_END].reshape(-1, 33, 4)
    pose[:, :, :3] *= factor
    out[:, :_POSE_END] = pose.reshape(-1, 33 * 4)

    # Scale hands and face (all x, y, z)
    out[:, _POSE_END:_FACE_END] *= factor

    return out


def frame_dropout(seq, drop_rate=0.05):
    """Randomly drop a fraction of frames (simulates frame skips / occlusion)."""
    T = seq.shape[0]
    keep_mask = np.random.rand(T) > drop_rate
    # Always keep at least 4 frames
    if keep_mask.sum() < 4:
        keep_mask[:4] = True
    return seq[keep_mask]


def landmark_dropout(seq, drop_prob=0.10):
    """
    Randomly zero out entire landmark groups per frame to simulate
    partial detection failure (e.g., hand not detected, face not visible).
    """
    out = seq.copy()
    T   = out.shape[0]

    # Define the four droppable groups and their slice ranges
    groups = [
        (0, _POSE_END),         # pose (132 values)
        (_POSE_END, _LH_END),   # left hand (63 values)
        (_LH_END, _RH_END),     # right hand (63 values)
        (_RH_END, _FACE_END),   # face (30 values)
    ]

    for start, end in groups:
        # Per-frame independent drop decision
        mask = np.random.rand(T) < drop_prob
        out[mask, start:end] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composite augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment(seq, p_noise=0.5, p_temporal=0.5, p_spatial=0.5,
            p_frame_drop=0.3, p_landmark_drop=0.3):
    """
    Apply a random subset of augmentations to a (T, 288) sequence.

    Each augmentation is applied independently with probability p_*.
    Some samples may get no augmentation, others may get all five.

    Parameters
    ----------
    seq : ndarray (T, 288) — already normalized

    Returns
    -------
    ndarray (T', 288) — augmented sequence (T' may differ due to temporal scaling)
    """
    out = seq.copy()

    if np.random.rand() < p_noise:
        out = add_gaussian_noise(out)

    if np.random.rand() < p_temporal:
        out = temporal_scale(out)

    if np.random.rand() < p_spatial:
        out = spatial_scale(out)

    if np.random.rand() < p_frame_drop:
        out = frame_dropout(out)

    if np.random.rand() < p_landmark_drop:
        out = landmark_dropout(out)

    return out
