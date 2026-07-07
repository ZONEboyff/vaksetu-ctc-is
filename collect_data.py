# -*- coding: utf-8 -*-
"""
collect_data.py — Autonomous webcam-based data collector.
Uses MediaPipe Tasks API (PoseLandmarker + HandLandmarker + FaceLandmarker).

Workflow
--------
  Phase 1 — Calibration take (manual):
    R  →  3-second countdown, then auto-start recording
    S  →  Stop & save; the duration of this take is remembered
    D  →  Discard current recording and go back to IDLE

  After Phase 1 the terminal asks:
    "How many more autonomous recordings?"

  Phase 2 — Autonomous loop (zero keyboard needed):
    For each take:
      REST (2 s)  →  COUNTDOWN (3 s)  →  RECORD (calibrated duration)  →  AUTO-SAVE
    D  →  Discard the current take (restarts from REST)
    Q  →  Stop early at any time

Run:
  python collect_data.py --sign please_carry_my_bag
  python collect_data.py --sign please
  python collect_data.py --none
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import argparse

from config import NONE_DIR, MIN_FRAMES_PER_SAMPLE, SIGNS
from capture_landmarks import extract_keypoints, download_model, FACE_KEY_INDICES

# ── Model paths ───────────────────────────────────────────────────────────────
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

# ── Timing constants ──────────────────────────────────────────────────────────
COUNTDOWN_SECS = 3.0   # seconds before each recording
REST_SECS      = 2.0   # seconds of rest between autonomous takes

# ── Hand skeleton connections ─────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Upper-body pose connections ───────────────────────────────────────────────
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
    """Draw pose + hand + face key landmarks onto frame."""
    # ── Pose ──
    if pose_result and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        draw_connections(frame, lms, POSE_CONNECTIONS, color=(0, 200, 100), thickness=2)
        draw_dots(frame, lms, color=(0, 255, 150), radius=3)

    # ── Hands ──
    if hand_result and hand_result.hand_landmarks:
        for idx, handedness_list in enumerate(hand_result.handedness):
            hand_type = handedness_list[0].category_name
            lms = hand_result.hand_landmarks[idx]
            if hand_type == "Left":
                conn_color, dot_color = (250, 44, 121), (255, 100, 180)
            else:
                conn_color, dot_color = (66, 117, 245), (120, 180, 255)
            draw_connections(frame, lms, HAND_CONNECTIONS, color=conn_color, thickness=2)
            draw_dots(frame, lms, color=dot_color, radius=5)

    # ── Face key points (cyan dots) ──
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
        cv2.putText(frame, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)


def draw_countdown_overlay(frame, remaining, subtitle="GET READY"):
    """Big centred countdown number with semi-transparent backdrop."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    digit  = str(int(remaining) + 1)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 7
    thick  = 14
    (tw, th), _ = cv2.getTextSize(digit, font, scale, thick)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, digit, (x + 3, y + 3), font, scale, (0,   0,   0), thick + 6, cv2.LINE_AA)
    cv2.putText(frame, digit, (x,     y    ), font, scale, (0, 220, 255), thick,     cv2.LINE_AA)

    sub_scale = 1.2
    (sw, _), _ = cv2.getTextSize(subtitle, font, sub_scale, 3)
    sx = (w - sw) // 2
    cv2.putText(frame, subtitle, (sx + 2, h - 55), font, sub_scale, (0,   0,   0), 5, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (sx,     h - 57), font, sub_scale, (255, 255, 255), 3, cv2.LINE_AA)


def draw_rest_overlay(frame, remaining, done, total):
    """Gentle 'REST' overlay between autonomous takes."""
    h, w = frame.shape[:2]
    msg = f"REST — next take in  {remaining:.1f}s   ({done}/{total} saved)"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x = (w - tw) // 2
    y = h - 30
    cv2.putText(frame, msg, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (x,     y    ), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (180, 180, 180), 2, cv2.LINE_AA)


def draw_progress_bar(frame, progress, done, total):
    """Horizontal recording progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_x, bar_y, bar_h = 20, h - 18, 10
    bar_w = w - 40
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * min(progress, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 100), -1)
    label = f"Take {done+1}/{total}"
    cv2.putText(frame, label, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def next_sample_path(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir)
                if f.startswith("sample_") and f.endswith(".npy")]
    indices = []
    for name in existing:
        try:
            indices.append(int(name.replace("sample_", "").replace(".npy", "")))
        except ValueError:
            pass
    next_idx = max(indices) + 1 if indices else 0
    return os.path.join(save_dir, f"sample_{next_idx}.npy")


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


def detect_frame(frame, pose_lm, hand_lm, face_lm):
    """Run all three landmarkers on a BGR frame. Returns (pose, hand, face, rgb_frame)."""
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return (pose_lm.detect(mp_img),
            hand_lm.detect(mp_img),
            face_lm.detect(mp_img),
            frame)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Manual calibration take
# ─────────────────────────────────────────────────────────────────────────────

def phase_calibration(cap, pose_lm, hand_lm, face_lm, mode_name, save_dir, saved_count):
    """
    Manual first recording with 3-second countdown.

    Returns
    -------
    (rec_duration_secs, saved_count)  on success
    (None, saved_count)               if user quit
    """
    state           = "idle"      # idle | countdown | recording
    buffer          = []
    countdown_start = 0.0
    rec_start       = 0.0

    print("\n── Phase 1: Calibration take ─────────────────────────────────────")
    print("  Press  R  for a 3-second countdown, then perform the sign.")
    print("  Press  S  when you're done.  D = discard.  Q = quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_r, hand_r, face_r, frame = detect_frame(frame, pose_lm, hand_lm, face_lm)
        draw_landmarks_on_frame(frame, pose_r, hand_r, face_r)

        now = time.time()

        # ── State transitions ──────────────────────────────────────────────
        if state == "countdown":
            remaining = COUNTDOWN_SECS - (now - countdown_start)
            if remaining <= 0:
                state     = "recording"
                buffer    = []
                rec_start = now
                print("[REC] Recording started …")
            else:
                draw_countdown_overlay(frame, remaining)

        elif state == "recording":
            buffer.append(extract_keypoints(pose_r, hand_r, face_r))
            cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 220), -1)

        # ── HUD ───────────────────────────────────────────────────────────
        if state == "idle":
            msg   = "Press R → 3s countdown → perform sign → press S to save"
            color = (200, 200, 200)
        elif state == "countdown":
            msg   = "GET READY …"
            color = (0, 200, 255)
        else:
            msg   = f"RECORDING ({len(buffer)} frames)  —  Press S to save"
            color = (0, 80, 220)

        overlay_text(frame, [
            f"Mode: {mode_name}  |  Saved: {saved_count}  |  {msg}",
            "R=Record   S=Save   D=Discard   Q=Quit",
        ], color=color)

        cv2.imshow(f"Data Collector [{mode_name}]", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            return None, saved_count

        if state == "idle" and key in (ord("r"), ord("R")):
            state           = "countdown"
            countdown_start = now
            print(f"[COUNTDOWN] Starting in {int(COUNTDOWN_SECS)}s …")

        elif state == "recording":
            if key in (ord("s"), ord("S")):
                duration = now - rec_start
                if len(buffer) >= MIN_FRAMES_PER_SAMPLE:
                    path = next_sample_path(save_dir)
                    arr  = np.array(buffer, dtype=np.float32)
                    np.save(path, arr)
                    saved_count += 1
                    print(f"[SAVE] {path}  shape={arr.shape}  "
                          f"duration={duration:.2f}s  total={saved_count}")
                    return duration, saved_count
                else:
                    print(f"[SKIP] Too short ({len(buffer)} frames). Try again.")
                    state  = "idle"
                    buffer = []

            elif key in (ord("d"), ord("D")):
                state  = "idle"
                buffer = []
                print("[DISCARD]")

    return None, saved_count


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Autonomous recording loop
# ─────────────────────────────────────────────────────────────────────────────

def phase_autonomous(cap, pose_lm, hand_lm, face_lm,
                     mode_name, save_dir, saved_count,
                     rec_duration, n_recordings):
    """
    Fully automated recording.
    Cycle: REST (2s) → COUNTDOWN (3s) → RECORD (rec_duration) → AUTO-SAVE

    D = discard current take (jumps back to REST)
    Q = stop early
    """
    state       = "rest"
    phase_start = time.time()
    buffer      = []
    done        = 0

    print(f"\n── Phase 2: Autonomous recording ─────────────────────────────────")
    print(f"  {n_recordings} takes  ×  {rec_duration:.2f}s each")
    print("  D = discard current take    Q = quit early")
    print("  (No other keyboard input needed — step back from the screen!)\n")

    while cap.isOpened() and done < n_recordings:
        ret, frame = cap.read()
        if not ret:
            break

        pose_r, hand_r, face_r, frame = detect_frame(frame, pose_lm, hand_lm, face_lm)
        draw_landmarks_on_frame(frame, pose_r, hand_r, face_r)

        now     = time.time()
        elapsed = now - phase_start

        # ── State machine ──────────────────────────────────────────────────
        if state == "rest":
            remaining = REST_SECS - elapsed
            if remaining <= 0:
                state       = "countdown"
                phase_start = now
                print(f"[AUTO] Take {done+1}/{n_recordings} — countdown …")
            else:
                draw_rest_overlay(frame, remaining, done, n_recordings)

        elif state == "countdown":
            remaining = COUNTDOWN_SECS - elapsed
            if remaining <= 0:
                state       = "recording"
                buffer      = []
                phase_start = now
                print(f"[AUTO REC] Take {done+1}/{n_recordings} started …")
            else:
                draw_countdown_overlay(frame, remaining,
                                       f"Take {done+1} / {n_recordings}")

        elif state == "recording":
            buffer.append(extract_keypoints(pose_r, hand_r, face_r))
            cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 220), -1)
            draw_progress_bar(frame, elapsed / rec_duration, done, n_recordings)

            if elapsed >= rec_duration:
                # ── Auto-save ──────────────────────────────────────────────
                if len(buffer) >= MIN_FRAMES_PER_SAMPLE:
                    path = next_sample_path(save_dir)
                    arr  = np.array(buffer, dtype=np.float32)
                    np.save(path, arr)
                    saved_count += 1
                    done        += 1
                    print(f"[AUTO SAVE] {path}  shape={arr.shape}  total={saved_count}")
                else:
                    print(f"[AUTO SKIP] Take {done+1} too short — retrying.")
                buffer      = []
                state       = "rest"
                phase_start = now

        # ── HUD ───────────────────────────────────────────────────────────
        if state == "rest":
            remaining = max(0.0, REST_SECS - elapsed)
            msg   = f"REST — next take in {remaining:.1f}s"
            color = (160, 160, 160)
        elif state == "countdown":
            msg   = f"GET READY — Take {done+1}/{n_recordings}"
            color = (0, 200, 255)
        else:
            msg   = f"RECORDING ({len(buffer)} frames) — Take {done+1}/{n_recordings}"
            color = (0, 80, 220)

        overlay_text(frame, [
            f"Mode: {mode_name}  |  Saved: {saved_count}  |  {msg}",
            f"AUTO MODE  |  Progress: {done}/{n_recordings}  |  D=Discard take   Q=Quit",
        ], color=color)

        cv2.imshow(f"Data Collector [{mode_name}]", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("[AUTO] Stopped early by user.")
            break

        if key in (ord("d"), ord("D")) and state == "recording":
            buffer      = []
            state       = "rest"
            phase_start = now
            print(f"[AUTO DISCARD] Take {done+1} discarded — restarting from rest.")

    print(f"\n[AUTO] Finished. {done}/{n_recordings} takes saved. Total: {saved_count}")
    return saved_count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sign", type=str, default="good_morning",
        choices=list(SIGNS.keys()),
        help="Which sign to record (default: good_morning)",
    )
    parser.add_argument("--none", action="store_true",
                        help="Record idle/negative samples into data/none/")
    args = parser.parse_args()

    if args.none:
        save_dir  = NONE_DIR
        mode_name = "IDLE / none"
    else:
        sign_cfg  = SIGNS[args.sign]
        save_dir  = sign_cfg["data_dir"]
        mode_name = sign_cfg["display"]
    os.makedirs(save_dir, exist_ok=True)

    # ── Download models ───────────────────────────────────────────────────
    download_model(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        POSE_MODEL)
    download_model(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL)
    download_model(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        FACE_MODEL)

    # ── Initialise landmarkers ────────────────────────────────────────────
    pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
            output_segmentation_masks=False))
    hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2))
    face_lm = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            num_faces=1))

    # ── Open camera ───────────────────────────────────────────────────────
    cap = None
    for idx, backend, label in [
        (0, cv2.CAP_DSHOW, "index=0 DirectShow"),
        (0, None,          "index=0 default"),
        (1, cv2.CAP_DSHOW, "index=1 DirectShow"),
        (1, None,          "index=1 default"),
    ]:
        print(f"[Camera] Trying {label} …", end="", flush=True)
        cap = try_open_camera(idx, backend)
        if cap is not None:
            print(" ready.")
            break
        print(" failed.")

    if cap is None:
        print("[ERROR] Could not open webcam.")
        return

    saved_count = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])

    print("=" * 60)
    print("  Sign Language Data Collector  (Autonomous Mode)")
    print(f"  Sign     : {mode_name}")
    print(f"  Save dir : {save_dir}")
    print(f"  Existing : {saved_count} samples")
    print("=" * 60)

    # ── Phase 1: manual calibration take ─────────────────────────────────
    duration, saved_count = phase_calibration(
        cap, pose_lm, hand_lm, face_lm, mode_name, save_dir, saved_count)

    if duration is None:
        print("[INFO] Session ended by user during calibration.")
    else:
        # ── Ask for autonomous count ──────────────────────────────────────
        print(f"\n✓  First recording done — duration: {duration:.2f}s")
        try:
            raw = input("How many more autonomous recordings? (0 to stop): ").strip()
            n   = max(0, int(raw))
        except (ValueError, EOFError, KeyboardInterrupt):
            n = 0
            print("[INFO] Invalid input — skipping autonomous phase.")

        # ── Phase 2: autonomous loop ──────────────────────────────────────
        if n > 0:
            saved_count = phase_autonomous(
                cap, pose_lm, hand_lm, face_lm,
                mode_name, save_dir, saved_count,
                duration, n)
        else:
            print("[INFO] Autonomous phase skipped.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    pose_lm.close()
    hand_lm.close()
    face_lm.close()
    print(f"\nSession complete. Total samples in '{save_dir}': {saved_count}")


if __name__ == "__main__":
    main()
