"""Paths and hyperparameter constants shared across the GNN pipeline.

Editing this single file propagates to training, evaluation, and graph building.
"""

from pathlib import Path

# ── Directory paths ────────────────────────────────────────────────────────────
# __file__ = squadhan/config.py, so .parent.parent = the repository root
GNN_ROOT = Path(__file__).resolve().parent.parent

# Raw K-League JSON data (KLEAGUE1/{season}/match/{game_id}/ layout)
RAW_DATA_DIR = GNN_ROOT / "raw-data"

# Output location of run_vaep.py
#   - vaep_oof.parquet : VAEP values for all matches (per-action offensive/defensive contribution)
#   - games.csv        : match metadata (game_id, home/away_team_id, result, etc.)
VAEP_OUTPUT_DIR = GNN_ROOT / "vaep" / "output"

# Folder of HeteroData .pt files saved by build_dataset.py
# 48D feature graphs go to a separate folder (preserving the original 276D graphs/ folder)
GRAPHS_DIR = GNN_ROOT / "outputs" / "graphs_48d"

# Folder for model weights saved whenever a new best val_loss is reached during training
CHECKPOINTS_DIR = GNN_ROOT / "outputs" / "checkpoints"

# Folder for evaluation result JSON files (AUC, LogLoss, Brier, ECE, etc.)
METRICS_DIR = GNN_ROOT / "outputs" / "metrics"


# ── K-League league identifiers ────────────────────────────────────────────────
# Numeric competition_id values used in the Bepro data
K1_COMPETITION_ID = 587   # K League 1 (first division)
K2_COMPETITION_ID = 588   # K League 2 (second division)

# Leagues this pipeline processes (excludes external data such as the J-League)
VALID_COMPETITION_IDS = {K1_COMPETITION_ID, K2_COMPETITION_ID}

# Seasons used (in 5-fold LOSO each season serves as the test set once)
SEASONS = [2021, 2022, 2023, 2024, 2025]


# ── Graph feature dimensions ───────────────────────────────────────────────────
# The pitch is split into 12 zones (see zones.py)
NUM_ZONES = 12

# Action types defined by the SPADL standard (per vaep/lib/.../spadl/config.py)
# type_ids 0~22 map to these actions
ACTION_TYPES = [
    "pass",              # 0: pass
    "cross",             # 1: cross
    "throw_in",          # 2: throw-in
    "freekick_crossed",  # 3: crossed free kick
    "freekick_short",    # 4: short free kick
    "corner_crossed",    # 5: crossed corner
    "corner_short",      # 6: short corner
    "take_on",           # 7: take-on (dribble past an opponent)
    "foul",              # 8: foul
    "tackle",            # 9: tackle
    "interception",      # 10: interception
    "shot",              # 11: shot
    "shot_penalty",      # 12: penalty kick
    "shot_freekick",     # 13: free-kick shot
    "keeper_save",       # 14: keeper save
    "keeper_claim",      # 15: keeper claim
    "keeper_punch",      # 16: keeper punch
    "keeper_pick_up",    # 17: keeper pick-up
    "clearance",         # 18: clearance
    "bad_touch",         # 19: bad touch (loss of ball control)
    "non_action",        # 20: non-action (bookkeeping)
    "dribble",           # 21: dribble (carry)
    "goalkick",          # 22: goal kick
]
NUM_ACTION_TYPES = len(ACTION_TYPES)   # 23

# Node feature dimension: redefined in the action-group section below (48D)

# Edge feature dimension: 12 slots, one per zone (B-scheme, see zones.py)
# B-scheme: the source player's vaep accumulates in the source zone, the destination player's vaep in the destination zone
EDGE_DIM = NUM_ZONES   # 12


# ── Action groups (276D → 48D feature compression) ─────────────────────────────
# The 23 action types are folded into 4 semantic groups → less sparsity
GROUP_MAP: dict[int, list[int]] = {
    0: [0, 1, 2, 3, 4, 5, 6, 22],    # passing/distribution: pass,cross,throw_in,freekick_*,corner_*,goalkick
    1: [11, 12, 13],                   # shooting: shot,shot_penalty,shot_freekick
    2: [8, 9, 10, 14, 15, 16, 18],    # defending: foul,tackle,interception,keeper_*,clearance
    3: [7, 19, 21],                    # ball carrying: take_on,bad_touch,dribble
    # dropped: 17 (keeper_pick_up, 0%), 20 (non_action, 0%) — absent from the data
}
NUM_GROUPS = 4

# Node feature dimension: action groups (4) × zones (12) = 48D  (was 276D → 48D)
NODE_DIM = NUM_GROUPS * NUM_ZONES   # 4 × 12 = 48


# ── GNN hyperparameters ────────────────────────────────────────────────────────
HIDDEN_CHANNELS = 64   # node embedding dimension
NUM_LAYERS = 2         # number of GNN convolution layers
NUM_HEADS = 4          # number of attention heads
DROPOUT = 0.3          # dropout rate

# Optimizer
LR = 1e-3              # Adam learning rate
WEIGHT_DECAY = 1e-4    # L2 regularization coefficient

# Training control
MAX_EPOCHS = 100       # max number of epochs
PATIENCE = 15          # epochs without val_loss improvement before early stopping
BATCH_SIZE = 32        # graphs per batch (passed to DataLoader)
