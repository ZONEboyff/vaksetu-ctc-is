"""
normalize.py — Shoulder-relative landmark normalization.

Makes features invariant to:
  - Person's position in the camera frame
  - Distance from camera
  - Body size differences

Applied at load time (training) and at predict time (inference).
The raw .npy files are NOT modified — normalization is a live transform.

Feature layout (288 values per frame):
  pose : 33 × (x, y, z, vis) = 132   → normalize x,y,z; keep vis
  lh   : 21 × (x, y, z)      =  63   → normalize all
  rh   : 21 × (x, y, z)      =  63   → normalize all
  face : 10 × (x, y, z)      =  30   → normalize all
"""

import numpy as np

# ── Pose landmark indices for the anchor points ──────────────────────────
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12

# ── Slice boundaries in the 288-dim vector ───────────────────────────────
_POSE_END = 33 * 4        # 132
_LH_END   = _POSE_END + 21 * 3   # 195
_RH_END   = _LH_END   + 21 * 3   # 258
_FACE_END = _RH_END   + 10 * 3   # 288


def _get_shoulder_anchor(pose_flat):
    """
    Extract shoulder centre (x, y, z) and shoulder width from the
    flattened pose vector (132 values: 33 landmarks × 4).

    Returns
    -------
    anchor : ndarray (3,)   — midpoint of left/right shoulder (x, y, z)
    scale  : float          — Euclidean distance between the two shoulders
    valid  : bool           — False if shoulders not detected (all zeros)
    """
    # Each landmark occupies 4 slots: x, y, z, visibility
    ls_idx = _LEFT_SHOULDER * 4
    rs_idx = _RIGHT_SHOULDER * 4

    ls = pose_flat[ls_idx: ls_idx + 3]     # left shoulder x, y, z
    rs = pose_flat[rs_idx: rs_idx + 3]     # right shoulder x, y, z

    # If both are near-zero the pose wasn't detected
    if np.linalg.norm(ls) < 1e-6 and np.linalg.norm(rs) < 1e-6:
        return np.zeros(3), 1.0, False

    anchor = (ls + rs) / 2.0
    scale  = np.linalg.norm(ls - rs)
    if scale < 1e-6:
        scale = 1.0   # avoid division by zero

    return anchor, scale, True


def normalize_frame(frame):
    """
    Normalize a single 288-dim feature vector in-place.

    Parameters
    ----------
    frame : ndarray (288,)

    Returns
    -------
    ndarray (288,) — normalized copy (original is not mutated)
    """
    out = frame.copy()
    pose_flat = out[:_POSE_END]

    anchor, scale, valid = _get_shoulder_anchor(pose_flat)
    if not valid:
        return out   # can't normalize without shoulders — return raw

    # ── Pose: normalize x, y, z but keep visibility ──────────────────────
    pose_reshaped = pose_flat.reshape(33, 4)
    pose_reshaped[:, :3] = (pose_reshaped[:, :3] - anchor) / scale
    out[:_POSE_END] = pose_reshaped.flatten()

    # ── Left hand ────────────────────────────────────────────────────────
    lh = out[_POSE_END:_LH_END].reshape(21, 3)
    # Only normalize if hand is detected (not all zeros)
    if np.any(lh):
        lh = (lh - anchor) / scale
        out[_POSE_END:_LH_END] = lh.flatten()

    # ── Right hand ───────────────────────────────────────────────────────
    rh = out[_LH_END:_RH_END].reshape(21, 3)
    if np.any(rh):
        rh = (rh - anchor) / scale
        out[_LH_END:_RH_END] = rh.flatten()

    # ── Face ─────────────────────────────────────────────────────────────
    face = out[_RH_END:_FACE_END].reshape(10, 3)
    if np.any(face):
        face = (face - anchor) / scale
        out[_RH_END:_FACE_END] = face.flatten()

    return out


def normalize_sequence(seq):
    """
    Normalize every frame in a (T, 288) sequence.

    Parameters
    ----------
    seq : ndarray (T, 288)

    Returns
    -------
    ndarray (T, 288) — normalized copy
    """
    return np.stack([normalize_frame(f) for f in seq])
