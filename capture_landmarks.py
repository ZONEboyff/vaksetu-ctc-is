import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# ---------------------------------------------------------------------------
# Key face landmark indices from MediaPipe FaceLandmarker (478-point model)
# Selected for ISL relevance: signs frequently reference chin, jaw, ears,
# nose, forehead, mouth corners, and lip centre.
# ---------------------------------------------------------------------------
FACE_KEY_INDICES = [
    152,   # chin tip
    234,   # jaw left
    454,   # jaw right
    93,    # left ear (cheek)
    323,   # right ear (cheek)
    4,     # nose tip
    10,    # forehead centre
    61,    # mouth left corner
    291,   # mouth right corner
    13,    # upper lip centre
]
# 10 landmarks × 3 (x, y, z) = 30 extra values
FACE_DIM = len(FACE_KEY_INDICES) * 3   # 30

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

def extract_keypoints(pose_result, hand_result, face_result=None):
    """
    Extracts x, y, z coordinates from MediaPipe Tasks API results and flattens them.
    If a body part isn't visible in the frame, it fills the array with zeros.

    Output layout (288 values total):
      pose  — 33 landmarks × 4 (x, y, z, visibility)  = 132
      lh    — 21 landmarks × 3 (x, y, z)               =  63
      rh    — 21 landmarks × 3 (x, y, z)               =  63
      face  — 10 key points  × 3 (x, y, z)             =  30
                                                 total  = 288
    """
    # ── Pose: 33 × 4 = 132 ──────────────────────────────────────────────
    if pose_result and pose_result.pose_landmarks:
        pose = np.array(
            [[res.x, res.y, res.z, res.visibility]
             for res in pose_result.pose_landmarks[0]]
        ).flatten()
    else:
        pose = np.zeros(33 * 4)

    # ── Hands: 21 × 3 per hand = 63 each ────────────────────────────────
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if hand_result and hand_result.hand_landmarks:
        for idx, handedness_list in enumerate(hand_result.handedness):
            hand_type = handedness_list[0].category_name   # "Left" or "Right"
            landmarks = hand_result.hand_landmarks[idx]
            arr = np.array([[res.x, res.y, res.z] for res in landmarks]).flatten()
            if hand_type == "Left":
                lh = arr
            elif hand_type == "Right":
                rh = arr

    # ── Face: 10 key points × 3 = 30 ────────────────────────────────────
    face = np.zeros(FACE_DIM)
    if face_result and face_result.face_landmarks:
        lms = face_result.face_landmarks[0]   # first detected face
        face = np.array(
            [[lms[i].x, lms[i].y, lms[i].z]
             for i in FACE_KEY_INDICES
             if i < len(lms)]
        ).flatten()
        # Guard: if some indices were out of range, pad to full size
        if len(face) < FACE_DIM:
            face = np.pad(face, (0, FACE_DIM - len(face)))

    # ── Concatenate: 132 + 63 + 63 + 30 = 288 ───────────────────────────
    return np.concatenate([pose, lh, rh, face])

def process_video(video_path, output_name):
    """Reads a video, extracts landmarks frame-by-frame, and saves the data."""

    pose_model_path = 'pose_landmarker_lite.task'
    hand_model_path = 'hand_landmarker.task'
    face_model_path = 'face_landmarker.task'

    # Download models if they don't exist
    download_model(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        pose_model_path)
    download_model(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        hand_model_path)
    download_model(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        face_model_path)

    # Initialize landmarkers
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            output_segmentation_masks=False))

    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2))

    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            num_faces=1))

    cap = cv2.VideoCapture(video_path)
    video_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = pose_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)

        keypoints = extract_keypoints(pose_result, hand_result, face_result)
        video_keypoints.append(keypoints)

    cap.release()
    pose_landmarker.close()
    hand_landmarker.close()
    face_landmarker.close()

    if not output_name.endswith('.npy'):
        output_name += '.npy'

    dir_name = os.path.dirname(output_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    np.save(output_name, np.array(video_keypoints))
    print(f"Saved {output_name}")

    return np.array(video_keypoints)

if __name__ == "__main__":
    # Example usage:
    # process_video(0, "output")
    pass