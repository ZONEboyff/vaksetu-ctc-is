# Vaksetu — CTC-Based Continuous Indian Sign Language Recognition

> A proof-of-concept **Continuous Sign Language Recognition (CSLR)** system for Indian Sign Language (ISL), powered by a **Conv1D + Bidirectional LSTM + Multi-Head Attention** architecture trained end-to-end with **CTC (Connectionist Temporal Classification) loss**.

---

## Table of Contents
- [Why CTC over Isolated SLR?](#why-ctc-over-isolated-slr)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [How to Add New Signs and Sentences](#how-to-add-new-signs-and-sentences)
- [Training](#training)
- [Inference (Live Webcam)](#inference-live-webcam)
- [Vocabulary & Grammar](#vocabulary--grammar)
- [Results](#results)

---

## Why CTC over Isolated SLR?

Most Sign Language Recognition systems classify a **fixed-length window** of video frames into a single word — effectively an isolated word recognizer. This approach has fundamental limitations in real-world use:

| Problem | Isolated SLR | CTC (This Project) |
|---|---|---|
| Speed variance | ❌ Struggles — assumes fixed frame count per sign | ✅ Handles naturally — aligns sequence dynamically |
| Co-articulation | ❌ Fails — blurry transitions between words cause errors | ✅ Explicitly learns inter-sign transitions |
| Sentence context | ❌ Stateless — each window is classified independently | ✅ BiLSTM carries memory of previous signs |
| Scalability | ❌ Adding N sentences = N new classes | ✅ Learns N tokens, decodes millions of combinations |
| Live feed noise | ❌ Predicts a sign even during idle/resting positions | ✅ Blank token handles non-sign frames |

---

## Architecture

The V2 model (`model_v2.py`) is a **hybrid architecture** combining:

```
Input (T × 288 landmarks per frame)
       ↓
Conv1D Front-End         — captures local hand shape patterns
       ↓
3-Layer Bidirectional LSTM — temporal sequence modeling (past + future context)
       ↓
Multi-Head Self-Attention  — global context across entire signing sequence
       ↓
FC Layer → CTC Decode      — outputs the gloss sequence (e.g. ["please", "carry", "my", "bag"])
```

**Key design decisions:**
- **BiLSTM over pure Transformer:** Transformers are data-hungry and fail on small datasets. BiLSTM has a sequential inductive bias that converges reliably with ~50 samples per class. Our Multi-Head Attention layer adds Transformer-level global reasoning on top.
- **CTC Loss over CE Loss:** No need for frame-level alignment labels. The model learns alignment automatically from (video, gloss-sequence) pairs.
- **Shoulder-relative normalization:** All landmarks are normalized relative to the shoulder midpoint and shoulder width, making the system invariant to the signer's position, distance from camera, and body size.
- **~3.2M parameters** — fast enough for real-time CPU/GPU inference.

---

## Project Structure

```
CTC/
├── config.py              # Central registry: vocabulary, SIGNS dict, data paths
├── capture_landmarks.py   # MediaPipe landmark extraction (pose + hands + face)
├── collect_data.py        # Interactive data collection script
├── normalize.py           # Shoulder-relative landmark normalization
├── augment.py             # Training-time data augmentation
├── dataset_v2.py          # PyTorch dataset with normalization + augmentation
├── model_v2.py            # Conv1D + BiLSTM + Attention model definition
├── train_v2.py            # Training loop with CosineAnnealingWarmRestarts
├── inference_v2.py        # Live webcam inference
├── requirements.txt       # Python dependencies
└── Vocabulary.csv         # ISL vocabulary with YouTube reference links
```

> **Legacy files** (`model.py`, `train.py`, `inference.py`, `dataset.py`) are V1 — kept for reference but not recommended for use.

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A CUDA-capable GPU is strongly recommended for training (CPU works but is slow)
- Webcam for data collection and inference

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download MediaPipe task models
Download the following files and place them in the `CTC/` root directory:

| File | Download Link |
|---|---|
| `face_landmarker.task` | [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models) |
| `hand_landmarker.task` | [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) |
| `pose_landmarker_lite.task` | [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) |

### 4. Get the dataset
The dataset (~87 MB of `.npy` files) is too large for GitHub.
Download the `data.zip` file shared separately and extract it into `CTC/data/`.

### 5. Get the model checkpoint
Download `model_v2_checkpoint.pt` shared separately and place it in `CTC/`.

---

## Dataset

Each sample is a `.npy` file of shape `(T, 288)`:
- **T** = number of frames (variable, typically 20–80)
- **288** = landmark features per frame:
  - Pose: 33 × 4 (x, y, z, visibility) = 132
  - Left hand: 21 × 3 (x, y, z) = 63
  - Right hand: 21 × 3 (x, y, z) = 63
  - Face (key points): 10 × 3 (x, y, z) = 30

Data is organized as:
```
data/
├── good/                    # ~42 samples of the isolated sign "good"
├── good_morning/            # ~41 samples of the sentence "good morning"
├── please_carry_my_bag/     # ~62 samples of the full sentence
├── we_monday_together_practice/
└── ...
```

---

## How to Add New Signs and Sentences

This is the core workflow for expanding the vocabulary. Follow these steps precisely.

### Step 1: Update `config.py`

Open `config.py` and do two things:

**A) Add new gloss tokens to `VOCAB`:**
```python
VOCAB = {
    0:  "<blank>",
    1:  "good",
    # ... existing entries ...
    23: "new_word",   # ← add here with the next available integer
}
```

**B) Register the new sign in `SIGNS`:**
```python
SIGNS = {
    # ... existing signs ...
    "new_word": {
        "display":  "new word",
        "glosses":  ["new_word"],
        "labels":   [GLOSS_TO_IDX["new_word"]],
        "data_dir": os.path.join(DATA_DIR, "new_word"),
    },
    # For a new sentence (ISL grammatical order!):
    "subject_object_verb": {
        "display":  "subject object verb",
        "glosses":  ["subject", "object", "verb"],
        "labels":   [GLOSS_TO_IDX["subject"], GLOSS_TO_IDX["object"], GLOSS_TO_IDX["verb"]],
        "data_dir": os.path.join(DATA_DIR, "subject_object_verb"),
    },
}
```

> ⚠️ **ISL Grammar Note:** ISL follows **Subject-Object-Verb (SOV)** word order. Question words (what, where, who, when) go **at the end** of the sentence. Always check the grammar before recording!

### Step 2: Collect Data

Run the collection script for each new entry:
```bash
python collect_data.py --sign new_word
```

**Recommended sample counts:**
- **Individual words:** 40 samples
- **Full sentences:** 60 samples (higher count helps the model learn co-articulation)

**Tips for best results:**
- Record in good, consistent lighting
- Keep your face and both shoulders in frame
- For individual words, sign clearly but at natural speed (not over-enunciated)
- For sentences, sign fluidly without pausing between words
- Record 5–10 samples of common **sub-sequences** (e.g., `"subject_object"` without the verb) to prevent the model from auto-completing hallucinated words

### Step 3: Train

```bash
python train_v2.py
```

Training runs for 250 epochs by default with Cosine Annealing Warm Restarts. A typical run takes 10–20 minutes on a mid-range GPU.

Output:
- `model_v2_checkpoint.pt` — best model (by validation loss)
- `training_log_v2.csv` — epoch-by-epoch loss/LR log

### Step 4: Test (Live Webcam)

```bash
python inference_v2.py
```

- Press **space** to start a recording
- Sign your sentence
- Press **space** again to stop — the decoded gloss sequence will appear on screen
- Press **Q** to quit

---

## Training

Key hyperparameters (set in `train_v2.py`):

| Param | Value | Notes |
|---|---|---|
| Epochs | 250 | With warm restarts |
| Batch size | 16 | |
| Optimizer | AdamW | weight_decay=1e-4 |
| Scheduler | CosineAnnealingWarmRestarts | T_0=50, T_mult=2 |
| CTC Blank | 0 | `<blank>` token |

---

## Vocabulary & Grammar

The current vocabulary contains **22 gloss tokens** (+ blank):

| Index | Word | Index | Word |
|---|---|---|---|
| 1 | good | 12 | book |
| 2 | morning | 13 | table |
| 3 | night | 14 | on |
| 4 | please | 15 | hello |
| 5 | carry | 16 | you |
| 6 | my | 17 | meet |
| 7 | bag | 18 | your |
| 8 | we | 19 | name |
| 9 | monday | 20 | what |
| 10 | together | 21 | house |
| 11 | practice | 22 | where |

**Trained sentences:**

| English | ISL Gloss Sequence |
|---|---|
| Good morning | `[GOOD, MORNING]` |
| Good night | `[GOOD, NIGHT]` |
| Please carry my bag | `[PLEASE, CARRY, MY, BAG]` |
| We practice together every Monday | `[WE, MONDAY, TOGETHER, PRACTICE]` |
| My book is on the table | `[MY, BOOK, TABLE, ON]` |
| Hello, nice to meet you | `[HELLO, YOU, MEET, GOOD]` |
| What is your name? | `[YOUR, NAME, WHAT]` |
| Where is your house? | `[YOUR, HOUSE, WHERE]` |

Reference YouTube links for each sign are in `Vocabulary.csv`.

---

## Results

- **Best validation loss (V2):** ~0.08 (CTC loss)
- The model correctly handles **novel combinations** of trained words (e.g., signs `"we monday"` or `"we together"` even though only `"we monday together practice"` was in training data)
- Real-time inference at ~20 FPS on GPU

---

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) — landmark extraction
- [PyTorch](https://pytorch.org/) — model training
- ISL reference videos from the [ISLRTC](https://islrtc.nic.in/) and Indian Sign Language community YouTube channels
