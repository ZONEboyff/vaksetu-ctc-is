"""
inference_v2.py — Run the v2 LSTM + CTC model with normalization.
Uses MediaPipe Tasks API (PoseLandmarker + HandLandmarker + FaceLandmarker).

Two modes
---------
  1. Webcam (default) : R=Record  S=Stop/decode  Q=Quit
  2. File             : python inference_v2.py --file path/to/video.mp4
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import VOCAB, BLANK_IDX, MIN_FRAMES_PER_SAMPLE, BLANK_RATIO_THRESHOLD
from model_v2 import build_model_v2
from capture_landmarks import extract_keypoints, download_model, FACE_KEY_INDICES
from normalize import normalize_sequence

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v2_checkpoint.pt")
POSE_MODEL    = "pose_landmarker_lite.task"
HAND_MODEL    = "hand_landmarker.task"
FACE_MODEL    = "face_landmarker.task"

# ── Skeleton connections ──────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_connections(frame, landmarks, connections, color, thickness=2):
    h, w = frame.shape[:2]
    for a, b in connections:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(frame, lm_to_px(landmarks[a], w, h),
                     lm_to_px(landmarks[b], w, h), color, thickness, cv2.LINE_AA)


def draw_dots(frame, landmarks, color, radius=4):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, lm_to_px(lm, w, h), radius, color, -1, cv2.LINE_AA)


def draw_landmarks_on_frame(frame, pose_result, hand_result, face_result=None):
    if pose_result and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        draw_connections(frame, lms, POSE_CONNECTIONS, color=(0, 200, 100))
        draw_dots(frame, lms, color=(0, 255, 150), radius=3)

    if hand_result and hand_result.hand_landmarks:
        for idx, handedness_list in enumerate(hand_result.handedness):
            hand_type = handedness_list[0].category_name
            lms = hand_result.hand_landmarks[idx]
            if hand_type == "Left":
                draw_connections(frame, lms, HAND_CONNECTIONS, color=(250, 44, 121))
                draw_dots(frame, lms, color=(255, 100, 180), radius=5)
            else:
                draw_connections(frame, lms, HAND_CONNECTIONS, color=(66, 117, 245))
                draw_dots(frame, lms, color=(120, 180, 255), radius=5)

    if face_result and face_result.face_landmarks:
        lms = face_result.face_landmarks[0]
        h, w = frame.shape[:2]
        for i in FACE_KEY_INDICES:
            if i < len(lms):
                cx, cy = int(lms[i].x * w), int(lms[i].y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 220, 255), -1, cv2.LINE_AA)


def overlay_text(frame, lines, start_y=30, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        y = start_y + i * 30
        cv2.putText(frame, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CTC greedy decode
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs_np, blank=BLANK_IDX):
    """log_probs_np : (T, C) — single sequence. Returns list of gloss strings."""
    indices     = log_probs_np.argmax(axis=-1)
    blank_ratio = (indices == blank).sum() / len(indices)
    if blank_ratio >= BLANK_RATIO_THRESHOLD:
        return []

    collapsed = []
    prev = None
    for tok in indices:
        if tok != prev:
            collapsed.append(tok)
        prev = tok
    return [VOCAB[t] for t in collapsed if t != blank]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_v2(device="cpu"):
    if not os.path.exists(V2_MODEL_PATH):
        print(f"[ERROR] No v2 checkpoint at '{V2_MODEL_PATH}'.")
        print("        Run  python train_v2.py  first.")
        sys.exit(1)
    model = build_model_v2(device)
    ckpt  = torch.load(V2_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[ModelV2] Loaded '{V2_MODEL_PATH}'  epoch={ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict_v2(model, sequence_np, device="cpu"):
    """
    sequence_np : (T, 288) — raw landmarks
    Normalizes, then runs through the model.
    """
    # Normalize before feeding to model
    norm_seq = normalize_sequence(sequence_np)
    t = torch.from_numpy(norm_seq).float().unsqueeze(1).to(device)   # (T, 1, 288)
    with torch.no_grad():
        log_probs = model(t)
    return ctc_greedy_decode(log_probs[:, 0, :].cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Build landmarkers
# ─────────────────────────────────────────────────────────────────────────────

def build_landmarkers():
    download_model(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        POSE_MODEL)
    download_model(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL)
    download_model(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        FACE_MODEL)

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
            output_segmentation_masks=False))
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2))
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            num_faces=1))
    return pose, hand, face


# ─────────────────────────────────────────────────────────────────────────────
# File inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_from_file(video_path, model, device):
    pose_lm, hand_lm, face_lm = build_landmarkers()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frames.append(extract_keypoints(
            pose_lm.detect(mp_image),
            hand_lm.detect(mp_image),
            face_lm.detect(mp_image)))

    cap.release()
    pose_lm.close()
    hand_lm.close()
    face_lm.close()

    if len(frames) < MIN_FRAMES_PER_SAMPLE:
        print(f"[WARN] Video too short ({len(frames)} frames).")
        return []

    return predict_v2(model, np.array(frames, dtype=np.float32), device)


# ─────────────────────────────────────────────────────────────────────────────
# Webcam inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_from_webcam(model, device):
    pose_lm, hand_lm, face_lm = build_landmarkers()

    def try_open_camera(index, backend=None):
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            return None
        for _ in range(60):
            ret, _ = cap.read()
            if ret:
                return cap
            time.sleep(0.033)
        cap.release()
        return None

    cap = None
    for idx, backend, label in [
        (0, cv2.CAP_DSHOW, "index=0 DirectShow"),
        (0, None,           "index=0 default"),
        (1, cv2.CAP_DSHOW, "index=1 DirectShow"),
        (1, None,           "index=1 default"),
    ]:
        print(f"[Camera] Trying {label} …", end="", flush=True)
        cap = try_open_camera(idx, backend)
        if cap is not None:
            print(" ready.")
            break
        print(" failed.")

    if cap is None:
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    recording   = False
    buffer      = []
    last_result = None

    print("=" * 55)
    print("  Inference V2 — webcam")
    print("  Normalization : ON")
    print("  R=Record  |  S=Decode  |  Q=Quit")
    print("=" * 55)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = pose_lm.detect(mp_image)
        hand_result = hand_lm.detect(mp_image)
        face_result = face_lm.detect(mp_image)

        draw_landmarks_on_frame(frame, pose_result, hand_result, face_result)

        if recording:
            buffer.append(extract_keypoints(pose_result, hand_result, face_result))
            cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 220), -1)

        result_str = " ".join(last_result) if last_result else "—"
        status     = f"RECORDING ({len(buffer)} frames)" if recording else "READY"
        color      = (80, 80, 220) if recording else (200, 200, 200)
        overlay_text(frame, [
            f"V2 | {status}",
            f"Prediction: {result_str}",
            "R=Record  S=Decode  Q=Quit",
        ], color=color)

        cv2.imshow("Sign Language Inference V2", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("r"), ord("R")):
            if not recording:
                recording   = True
                buffer      = []
                last_result = None
                print("[REC] Recording …")

        elif key in (ord("s"), ord("S")):
            if recording:
                recording = False
                if len(buffer) >= MIN_FRAMES_PER_SAMPLE:
                    glosses     = predict_v2(model, np.array(buffer, dtype=np.float32), device)
                    last_result = glosses if glosses else ["<no sign>"]
                    print(f"[RESULT] {' '.join(last_result)}")
                else:
                    print(f"[WARN] Too short ({len(buffer)} frames).")
                buffer = []

        elif key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_lm.close()
    hand_lm.close()
    face_lm.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LSTM + CTC Sign Language Inference V2")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a video file. Omit to use webcam.")
    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model_v2(device)

    if args.file:
        glosses = infer_from_file(args.file, model, device)
        print(f"\n[RESULT] {' '.join(glosses) if glosses else '<no prediction>'}")
    else:
        infer_from_webcam(model, device)


if __name__ == "__main__":
    main()
