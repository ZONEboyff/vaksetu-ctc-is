"""
config.py — Shared configuration for the LSTM + CTC sign language pipeline.
All scripts import from here so there is a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Vocabulary
# Index 0 is ALWAYS reserved for the CTC blank token.
# ---------------------------------------------------------------------------
VOCAB = {
    0: "<blank>",
    1: "good",
    2: "morning",
    3: "night",
    4: "please",
    5: "carry",
    6: "my",
    7: "bag",
    8: "we",
    9: "monday",
    10: "together",
    11: "practice",
    12: "book",
    13: "table",
    14: "on",
    15: "hello",
    16: "you",
    17: "meet",
    18: "your",
    19: "name",
    20: "what",
    21: "house",
    22: "where",
    23: "i",
    24: "sign_language",
    25: "learn",
    26: "understand",
    27: "thank_you",
    28: "bye",
    29: "doctor",
    30: "him",
    31: "help",
}

# Reverse lookup: gloss string → integer label
GLOSS_TO_IDX = {v: k for k, v in VOCAB.items() if k != 0}

BLANK_IDX   = 0
VOCAB_SIZE  = len(VOCAB)   # 8  (blank + good + morning + night + please + carry + my + bag)

# ---------------------------------------------------------------------------
# Multi-sign registry
# Add new signs here — collect_data.py, dataset.py, and train.py all read
# from this dict automatically.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SIGNS = {
    "good_morning": {
        "display":  "good morning",
        "glosses":  ["good", "morning"],
        "labels":   [GLOSS_TO_IDX["good"], GLOSS_TO_IDX["morning"]],   # [1, 2]
        "data_dir": os.path.join(DATA_DIR, "good_morning"),
    },
    "good_night": {
        "display":  "good night",
        "glosses":  ["good", "night"],
        "labels":   [GLOSS_TO_IDX["good"], GLOSS_TO_IDX["night"]],      # [1, 3]
        "data_dir": os.path.join(DATA_DIR, "good_night"),
    },

    # ── Individual words (good / morning / night) ────────────────────────
    "good": {
        "display":  "good",
        "glosses":  ["good"],
        "labels":   [GLOSS_TO_IDX["good"]],                              # [1]
        "data_dir": os.path.join(DATA_DIR, "good"),
    },
    "morning": {
        "display":  "morning",
        "glosses":  ["morning"],
        "labels":   [GLOSS_TO_IDX["morning"]],                           # [2]
        "data_dir": os.path.join(DATA_DIR, "morning"),
    },
    "night": {
        "display":  "night",
        "glosses":  ["night"],
        "labels":   [GLOSS_TO_IDX["night"]],                             # [3]
        "data_dir": os.path.join(DATA_DIR, "night"),
    },

    # ── Please carry my bag ──────────────────────────────────────────────
    "please_carry_my_bag": {
        "display":  "please carry my bag",
        "glosses":  ["please", "carry", "my", "bag"],
        "labels":   [GLOSS_TO_IDX["please"], GLOSS_TO_IDX["carry"],
                     GLOSS_TO_IDX["my"],     GLOSS_TO_IDX["bag"]],      # [4, 5, 6, 7]
        "data_dir": os.path.join(DATA_DIR, "please_carry_my_bag"),
    },

    # ── Individual words (for single-sign recognition) ───────────────────
    "please": {
        "display":  "please",
        "glosses":  ["please"],
        "labels":   [GLOSS_TO_IDX["please"]],                            # [4]
        "data_dir": os.path.join(DATA_DIR, "please"),
    },
    "carry": {
        "display":  "carry",
        "glosses":  ["carry"],
        "labels":   [GLOSS_TO_IDX["carry"]],                             # [5]
        "data_dir": os.path.join(DATA_DIR, "carry"),
    },
    "my": {
        "display":  "my",
        "glosses":  ["my"],
        "labels":   [GLOSS_TO_IDX["my"]],                                # [6]
        "data_dir": os.path.join(DATA_DIR, "my"),
    },
    "bag": {
        "display":  "bag",
        "glosses":  ["bag"],
        "labels":   [GLOSS_TO_IDX["bag"]],                               # [7]
        "data_dir": os.path.join(DATA_DIR, "bag"),
    },

    # ── We Monday Together Practice ──────────────────────────────────────
    "we_monday_together_practice": {
        "display":  "we monday together practice",
        "glosses":  ["we", "monday", "together", "practice"],
        "labels":   [GLOSS_TO_IDX["we"], GLOSS_TO_IDX["monday"],
                     GLOSS_TO_IDX["together"], GLOSS_TO_IDX["practice"]],
        "data_dir": os.path.join(DATA_DIR, "we_monday_together_practice"),
    },
    "we_monday_together": {
        "display":  "we monday together",
        "glosses":  ["we", "monday", "together"],
        "labels":   [GLOSS_TO_IDX["we"], GLOSS_TO_IDX["monday"], GLOSS_TO_IDX["together"]],
        "data_dir": os.path.join(DATA_DIR, "we_monday_together"),
    },

    # ── Individual words (for single-sign recognition) ───────────────────
    "we": {
        "display":  "we",
        "glosses":  ["we"],
        "labels":   [GLOSS_TO_IDX["we"]],
        "data_dir": os.path.join(DATA_DIR, "we"),
    },
    "monday": {
        "display":  "monday",
        "glosses":  ["monday"],
        "labels":   [GLOSS_TO_IDX["monday"]],
        "data_dir": os.path.join(DATA_DIR, "monday"),
    },
    "together": {
        "display":  "together",
        "glosses":  ["together"],
        "labels":   [GLOSS_TO_IDX["together"]],
        "data_dir": os.path.join(DATA_DIR, "together"),
    },
    "practice": {
        "display":  "practice",
        "glosses":  ["practice"],
        "labels":   [GLOSS_TO_IDX["practice"]],
        "data_dir": os.path.join(DATA_DIR, "practice"),
    },

    # ── My Book Table On ─────────────────────────────────────────────────
    "my_book_table_on": {
        "display":  "my book table on",
        "glosses":  ["my", "book", "table", "on"],
        "labels":   [GLOSS_TO_IDX["my"], GLOSS_TO_IDX["book"],
                     GLOSS_TO_IDX["table"], GLOSS_TO_IDX["on"]],
        "data_dir": os.path.join(DATA_DIR, "my_book_table_on"),
    },

    "book": {
        "display":  "book",
        "glosses":  ["book"],
        "labels":   [GLOSS_TO_IDX["book"]],
        "data_dir": os.path.join(DATA_DIR, "book"),
    },
    "table": {
        "display":  "table",
        "glosses":  ["table"],
        "labels":   [GLOSS_TO_IDX["table"]],
        "data_dir": os.path.join(DATA_DIR, "table"),
    },
    "on": {
        "display":  "on",
        "glosses":  ["on"],
        "labels":   [GLOSS_TO_IDX["on"]],
        "data_dir": os.path.join(DATA_DIR, "on"),
    },

    # ── Hello You Meet Good ──────────────────────────────────────────────
    "hello_you_meet_good": {
        "display":  "hello you meet good",
        "glosses":  ["hello", "you", "meet", "good"],
        "labels":   [GLOSS_TO_IDX["hello"], GLOSS_TO_IDX["you"],
                     GLOSS_TO_IDX["meet"], GLOSS_TO_IDX["good"]],
        "data_dir": os.path.join(DATA_DIR, "hello_you_meet_good"),
    },

    "hello": {
        "display":  "hello",
        "glosses":  ["hello"],
        "labels":   [GLOSS_TO_IDX["hello"]],
        "data_dir": os.path.join(DATA_DIR, "hello"),
    },
    "you": {
        "display":  "you",
        "glosses":  ["you"],
        "labels":   [GLOSS_TO_IDX["you"]],
        "data_dir": os.path.join(DATA_DIR, "you"),
    },
    "meet": {
        "display":  "meet",
        "glosses":  ["meet"],
        "labels":   [GLOSS_TO_IDX["meet"]],
        "data_dir": os.path.join(DATA_DIR, "meet"),
    },

    # ── Your Name What ───────────────────────────────────────────────────
    "your_name_what": {
        "display":  "your name what",
        "glosses":  ["your", "name", "what"],
        "labels":   [GLOSS_TO_IDX["your"], GLOSS_TO_IDX["name"],
                     GLOSS_TO_IDX["what"]],
        "data_dir": os.path.join(DATA_DIR, "your_name_what"),
    },

    "your": {
        "display":  "your",
        "glosses":  ["your"],
        "labels":   [GLOSS_TO_IDX["your"]],
        "data_dir": os.path.join(DATA_DIR, "your"),
    },
    "name": {
        "display":  "name",
        "glosses":  ["name"],
        "labels":   [GLOSS_TO_IDX["name"]],
        "data_dir": os.path.join(DATA_DIR, "name"),
    },
    "what": {
        "display":  "what",
        "glosses":  ["what"],
        "labels":   [GLOSS_TO_IDX["what"]],
        "data_dir": os.path.join(DATA_DIR, "what"),
    },

    # ── Your House Where ─────────────────────────────────────────────────
    "your_house_where": {
        "display":  "your house where",
        "glosses":  ["your", "house", "where"],
        "labels":   [GLOSS_TO_IDX["your"], GLOSS_TO_IDX["house"],
                     GLOSS_TO_IDX["where"]],
        "data_dir": os.path.join(DATA_DIR, "your_house_where"),
    },

    "house": {
        "display":  "house",
        "glosses":  ["house"],
        "labels":   [GLOSS_TO_IDX["house"]],
        "data_dir": os.path.join(DATA_DIR, "house"),
    },
    "where": {
        "display":  "where",
        "glosses":  ["where"],
        "labels":   [GLOSS_TO_IDX["where"]],
        "data_dir": os.path.join(DATA_DIR, "where"),
    },

    # ── I Sign Language Learn ────────────────────────────────────────────
    "i_sign_language_learn": {
        "display":  "i sign language learn",
        "glosses":  ["i", "sign_language", "learn"],
        "labels":   [GLOSS_TO_IDX["i"], GLOSS_TO_IDX["sign_language"], GLOSS_TO_IDX["learn"]],
        "data_dir": os.path.join(DATA_DIR, "i_sign_language_learn"),
    },
    
    # ── You Understand ───────────────────────────────────────────────────
    "you_understand": {
        "display":  "you understand",
        "glosses":  ["you", "understand"],
        "labels":   [GLOSS_TO_IDX["you"], GLOSS_TO_IDX["understand"]],
        "data_dir": os.path.join(DATA_DIR, "you_understand"),
    },

    # ── Thank You Bye ────────────────────────────────────────────────────
    "thank_you_bye": {
        "display":  "thank you bye",
        "glosses":  ["thank_you", "bye"],
        "labels":   [GLOSS_TO_IDX["thank_you"], GLOSS_TO_IDX["bye"]],
        "data_dir": os.path.join(DATA_DIR, "thank_you_bye"),
    },

    # ── Doctor Him Help Please ───────────────────────────────────────────
    "doctor_him_help_please": {
        "display":  "doctor him help please",
        "glosses":  ["doctor", "him", "help", "please"],
        "labels":   [GLOSS_TO_IDX["doctor"], GLOSS_TO_IDX["him"],
                     GLOSS_TO_IDX["help"], GLOSS_TO_IDX["please"]],
        "data_dir": os.path.join(DATA_DIR, "doctor_him_help_please"),
    },

    "i": {
        "display":  "i",
        "glosses":  ["i"],
        "labels":   [GLOSS_TO_IDX["i"]],
        "data_dir": os.path.join(DATA_DIR, "i"),
    },
    "sign_language": {
        "display":  "sign language",
        "glosses":  ["sign_language"],
        "labels":   [GLOSS_TO_IDX["sign_language"]],
        "data_dir": os.path.join(DATA_DIR, "sign_language"),
    },
    "learn": {
        "display":  "learn",
        "glosses":  ["learn"],
        "labels":   [GLOSS_TO_IDX["learn"]],
        "data_dir": os.path.join(DATA_DIR, "learn"),
    },
    "understand": {
        "display":  "understand",
        "glosses":  ["understand"],
        "labels":   [GLOSS_TO_IDX["understand"]],
        "data_dir": os.path.join(DATA_DIR, "understand"),
    },
    "thank_you": {
        "display":  "thank you",
        "glosses":  ["thank_you"],
        "labels":   [GLOSS_TO_IDX["thank_you"]],
        "data_dir": os.path.join(DATA_DIR, "thank_you"),
    },
    "bye": {
        "display":  "bye",
        "glosses":  ["bye"],
        "labels":   [GLOSS_TO_IDX["bye"]],
        "data_dir": os.path.join(DATA_DIR, "bye"),
    },
    "doctor": {
        "display":  "doctor",
        "glosses":  ["doctor"],
        "labels":   [GLOSS_TO_IDX["doctor"]],
        "data_dir": os.path.join(DATA_DIR, "doctor"),
    },
    "him": {
        "display":  "him",
        "glosses":  ["him"],
        "labels":   [GLOSS_TO_IDX["him"]],
        "data_dir": os.path.join(DATA_DIR, "him"),
    },
    "help": {
        "display":  "help",
        "glosses":  ["help"],
        "labels":   [GLOSS_TO_IDX["help"]],
        "data_dir": os.path.join(DATA_DIR, "help"),
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by dataset.py, inference.py, etc.)
# ---------------------------------------------------------------------------
SENTENCE        = "good_morning"
TARGET_GLOSSES  = SIGNS["good_morning"]["glosses"]
TARGET_LABELS   = SIGNS["good_morning"]["labels"]

SENTENCE_DIR    = SIGNS["good_morning"]["data_dir"]   # data/good_morning/
NONE_DIR        = os.path.join(DATA_DIR, "none")       # data/none/ — negative/idle samples
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model_checkpoint.pt")

# Empty label tensor for "none" / idle samples (CTC target = all blanks)
NONE_LABEL      = []   # empty list → target_length = 0

# ---------------------------------------------------------------------------
# Landmark / feature configuration
# MediaPipe: 33 pose × 4 + 21 left-hand × 3 + 21 right-hand × 3 = 258
#            + 10 face key-points × 3 (chin, jaw, ears, nose, forehead, mouth) = 30
# ---------------------------------------------------------------------------
FEATURE_DIM = 288   # 258 pose/hand + 30 face

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 4
NUM_EPOCHS    = 150
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.2      # fraction of data used for validation
SEED          = 42

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# If more than this fraction of greedy-decoded frames are blank,
# treat the recording as "no sign" and output nothing.
BLANK_RATIO_THRESHOLD = 0.98

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
MIN_FRAMES_PER_SAMPLE = 10   # discard accidental tiny recordings
