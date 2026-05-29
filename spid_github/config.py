"""SPID (Split-based Prompt Injection Detector) — configuration."""

import random
import numpy as np
import torch

# ── reproducibility ──
SEED = 42

# ── model ──
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256

# ── training ──
CLASS_WEIGHTS_LIST = [2.0, 1.0]   # [safe, unsafe] — penalise FP more
LABEL_SMOOTHING = 0.1

# round 1
R1_EPOCHS = 3
R1_LR = 1e-5
R1_BATCH = 8
R1_GRAD_ACCUM = 2       # effective batch = 16
R1_WARMUP = 0.1

# round 2 (hard-negative mining)
R2_EPOCHS = 1
R2_LR = 5e-6
R2_BATCH = 8
R2_GRAD_ACCUM = 2
R2_WARMUP = 0.05

WEIGHT_DECAY = 0.01

# ── evaluation ──
TARGET_PRECISION = 0.90
MIN_RECALL = 0.65

# ── seed everything ──
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
